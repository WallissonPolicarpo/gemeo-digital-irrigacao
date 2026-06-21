"""
firebase_writer.py — Gêmeo Digital de Irrigação
================================================
Lê o dataset futuro linha a linha, executa o modelo de IA localmente
e publica cada leitura no Firebase Realtime Database em tempo real.

Pré-requisitos
--------------
    pip install firebase-admin pandas numpy scikit-learn xgboost joblib

Configuração (uma vez só)
-------------------------
1. Acesse https://console.firebase.google.com
2. Crie um projeto (ex: "gemeo-digital-irrigacao")
3. Vá em Project Settings → Service Accounts → Generate new private key
4. Salve o JSON baixado como  serviceAccountKey.json  nesta mesma pasta
5. Em Realtime Database → Criar banco de dados → Modo de teste (30 dias)
6. Copie a URL do banco (ex: https://gemeo-digital-irrigacao-default-rtdb.firebaseio.com)
7. Substitua DATABASE_URL abaixo pela sua URL

Uso
---
    # Demo rápida (2 leituras por segundo, ~5 min para 14 dias de dados)
    python firebase_writer.py --rapido 2

    # Demo média (1 leitura por segundo)
    python firebase_writer.py --rapido 1

    # Simula tempo real (1 leitura a cada 30 minutos — uso em produção)
    python firebase_writer.py --real

    # Modo offline: imprime o JSON sem enviar ao Firebase (para testes)
    python firebase_writer.py --offline --rapido 5
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Configuração — EDITE AQUI ─────────────────────────────────────────────────

DATABASE_URL     = "https://SEU-PROJETO-default-rtdb.firebaseio.com"
CHAVE_JSON       = "serviceAccountKey.json"
CAMINHO_DB       = "/gemeo_digital"       # nó raiz no Firebase
CAMINHO_HISTORICO = "/gemeo_digital/historico"  # lista de leituras anteriores
MAX_HISTORICO    = 100                    # últimas N leituras guardadas no histórico

# Caminhos dos arquivos locais (ajuste se necessário)
DATASET_FUTURO   = Path(__file__).parent / "dataset_futuro.csv"
MODELO_PKL       = Path(__file__).parent / "modelo_irrigacao.pkl"
SCALER_PKL       = Path(__file__).parent / "scaler_irrigacao.pkl"
FEATURES_PKL     = Path(__file__).parent / "feature_cols.pkl"

# ── Constantes do pipeline de features ───────────────────────────────────────

WINDOW = 12   # janela deslizante de 6h (12 leituras × 30min)
COLUNAS_SENSOR = [
    'temp_ar_C', 'umid_ar_pct', 'temp_solo_C', 'umid_solo_pct',
    'radiacao_solar_Wm2', 'vento_ms', 'eto_penman_monteith_mm',
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _urgencia(horas: float) -> str:
    if horas <= 4:   return "URGENTE"
    if horas <= 12:  return "ATENCAO"
    if horas <= 24:  return "PROGRAMAR"
    return "OK"


def _nivel_umidade(pct: float) -> str:
    """Classifica umidade do solo em faixas agronômicas."""
    if pct < 15:  return "CRITICO"
    if pct < 22:  return "BAIXO"
    if pct < 35:  return "ADEQUADO"
    if pct < 45:  return "ELEVADO"
    return "SATURADO"


def _montar_payload(row: pd.Series, pred: np.ndarray,
                    indice: int, total: int, modo: str) -> dict:
    """
    Constrói o objeto JSON que será enviado ao Firebase.
    Estrutura pensada para o dashboard consumir diretamente.
    """
    h = float(np.clip(pred[0], 0, 72))
    v = float(np.clip(pred[1], 0, 50))
    c = float(np.clip(pred[2], 0, 1))

    return {
        # Identificação temporal
        "timestamp": str(row["timestamp"]),
        "atualizado_em": datetime.now().isoformat(timespec="seconds"),

        # Leituras dos 4 sensores
        "sensores": {
            "temp_ar_C":          round(float(row["temp_ar_C"]),   1),
            "umid_ar_pct":        round(float(row["umid_ar_pct"]), 1),
            "temp_solo_C":        round(float(row["temp_solo_C"]), 1),
            "umid_solo_pct":      round(float(row["umid_solo_pct"]), 1),
            "radiacao_solar_Wm2": round(float(row["radiacao_solar_Wm2"]), 1),
            "vento_ms":           round(float(row["vento_ms"]), 2),
            "nivel_umidade_solo": _nivel_umidade(float(row["umid_solo_pct"])),
        },

        # ETo calculada localmente via Penman-Monteith
        "penman_monteith": {
            "eto_mm_30min":  round(float(row["eto_penman_monteith_mm"]), 4),
            "eto_mm_dia_eq": round(float(row["eto_penman_monteith_mm"]) * 48, 3),
        },

        # Saída do modelo de IA
        "predicao": {
            "horas_ate_irrigacao": round(h, 1),
            "volume_L_m2":         round(v, 2),
            "confianca_pct":       round(c * 100, 1),
            "urgencia":            _urgencia(h),
            # Texto pronto para exibir no dashboard
            "resumo": (
                f"Irrigar em {h:.0f}h | "
                f"{v:.1f} L/m² | "
                f"Confiança {c*100:.0f}%"
            ),
        },

        # Metadados da simulação (úteis para debug e apresentação)
        "meta": {
            "indice_linha":   indice,
            "total_linhas":   total,
            "progresso_pct":  round(indice / total * 100, 1),
            "modo":           modo,
            "modelo":         "XGBoost MultiOutputRegressor",
        },
    }


# ── Engenharia de features (idêntica ao treinamento) ─────────────────────────

def _preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica exatamente o mesmo pipeline de features usado no treinamento.
    IMPORTANTE: qualquer alteração aqui deve ser espelhada em modelo_irrigacao.py.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Janela deslizante 6h para cada sensor
    for col in COLUNAS_SENSOR:
        df[f"{col}_mean6h"] = df[col].rolling(WINDOW, min_periods=1).mean()
        df[f"{col}_std6h"]  = df[col].rolling(WINDOW, min_periods=1).std().fillna(0)
        df[f"{col}_min6h"]  = df[col].rolling(WINDOW, min_periods=1).min()
        df[f"{col}_max6h"]  = df[col].rolling(WINDOW, min_periods=1).max()

    # Codificação cíclica de hora e dia do ano
    df["hora"]    = df["timestamp"].dt.hour
    df["doy"]     = df["timestamp"].dt.dayofyear
    df["hora_sin"] = np.sin(2 * math.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * math.pi * df["hora"] / 24)
    df["doy_sin"]  = np.sin(2 * math.pi * df["doy"] / 365)
    df["doy_cos"]  = np.cos(2 * math.pi * df["doy"] / 365)

    # ETo acumulada nas últimas 24h (proxy de déficit hídrico)
    df["eto_acum_24h"] = (
        df["eto_penman_monteith_mm"].rolling(48, min_periods=1).sum()
    )

    return df


# ── Inicialização Firebase ────────────────────────────────────────────────────

def _init_firebase():
    """
    Inicializa a conexão com o Firebase.
    Retorna (ref_principal, ref_historico) ou levanta RuntimeError se falhar.
    """
    try:
        import firebase_admin
        from firebase_admin import credentials, db as firebase_db
    except ImportError:
        raise RuntimeError(
            "Pacote firebase-admin não encontrado.\n"
            "Execute:  pip install firebase-admin"
        )

    chave = Path(CHAVE_JSON)
    if not chave.exists():
        raise RuntimeError(
            f"Arquivo de credenciais não encontrado: {chave.resolve()}\n"
            "Siga as instruções no início deste arquivo para gerar o serviceAccountKey.json."
        )

    if not firebase_admin._apps:   # evita reinicializar se já conectado
        cred = credentials.Certificate(str(chave))
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

    ref_principal  = firebase_db.reference(CAMINHO_DB)
    ref_historico  = firebase_db.reference(CAMINHO_HISTORICO)
    return ref_principal, ref_historico


# ── Loop principal ────────────────────────────────────────────────────────────

def executar(modo: str = "rapido", taxa: float = 2.0, offline: bool = False):
    """
    Parâmetros
    ----------
    modo    : "rapido" ou "real"
    taxa    : leituras por segundo (modo rápido)
    offline : se True, imprime o JSON mas não conecta ao Firebase
    """

    # ── Valida arquivos locais ────────────────────────────────────────────────
    for arq in [DATASET_FUTURO, MODELO_PKL, SCALER_PKL, FEATURES_PKL]:
        if not arq.exists():
            print(f"[ERRO] Arquivo não encontrado: {arq}", file=sys.stderr)
            sys.exit(1)

    # ── Carrega artefatos ─────────────────────────────────────────────────────
    print("[INFO] Carregando modelo e dataset...", file=sys.stderr)
    modelo       = joblib.load(MODELO_PKL)
    scaler       = joblib.load(SCALER_PKL)
    feature_cols = joblib.load(FEATURES_PKL)

    df_raw = pd.read_csv(DATASET_FUTURO, parse_dates=["timestamp"])
    df     = _preparar_features(df_raw)
    total  = len(df)
    print(f"[INFO] {total} leituras carregadas.", file=sys.stderr)

    # ── Conecta ao Firebase (se não for offline) ──────────────────────────────
    ref_principal = ref_historico = None
    if not offline:
        print("[INFO] Conectando ao Firebase...", file=sys.stderr)
        try:
            ref_principal, ref_historico = _init_firebase()
            print("[INFO] Firebase conectado.", file=sys.stderr)
        except RuntimeError as e:
            print(f"[ERRO] {e}", file=sys.stderr)
            sys.exit(1)

    # ── Intervalo entre leituras ──────────────────────────────────────────────
    intervalo = (1800.0 if modo == "real" else 1.0 / taxa)

    print(f"[INFO] Iniciando em 2s... (modo={modo}, intervalo={intervalo:.2f}s)\n",
          file=sys.stderr)
    time.sleep(2)

    historico_local: list[dict] = []   # buffer local do histórico

    for indice, (_, row) in enumerate(df.iterrows()):

        # ── Inferência ────────────────────────────────────────────────────────
        X   = row[feature_cols].fillna(0).values.reshape(1, -1)
        X_s = scaler.transform(X)
        pred = modelo.predict(X_s)[0]

        payload = _montar_payload(row, pred, indice, total, modo)

        # ── Publica ou imprime ────────────────────────────────────────────────
        if offline:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            try:
                # Atualiza o nó principal (leitura atual — sobrescreve)
                ref_principal.set(payload)

                # Mantém histórico das últimas MAX_HISTORICO leituras
                historico_local.append({
                    "t":  payload["timestamp"],
                    "ta": payload["sensores"]["temp_ar_C"],
                    "ua": payload["sensores"]["umid_ar_pct"],
                    "ts": payload["sensores"]["temp_solo_C"],
                    "us": payload["sensores"]["umid_solo_pct"],
                    "h":  payload["predicao"]["horas_ate_irrigacao"],
                    "v":  payload["predicao"]["volume_L_m2"],
                    "c":  payload["predicao"]["confianca_pct"],
                })
                if len(historico_local) > MAX_HISTORICO:
                    historico_local.pop(0)
                ref_historico.set(historico_local)

            except Exception as e:
                # Falha de rede não interrompe a demo — apenas loga
                print(f"[AVISO] Falha ao escrever no Firebase: {e}", file=sys.stderr)

        # ── Log no terminal ───────────────────────────────────────────────────
        s    = payload["sensores"]
        pred = payload["predicao"]
        urgencia_str = {
            "URGENTE":   "🔴 URGENTE",
            "ATENCAO":   "🟡 ATENÇÃO",
            "PROGRAMAR": "🟢 PROGRAMAR",
            "OK":        "✅ OK",
        }.get(pred["urgencia"], pred["urgencia"])

        print(
            f"[{payload['timestamp']}]  "
            f"T_ar={s['temp_ar_C']:5.1f}°C  "
            f"UR_ar={s['umid_ar_pct']:5.1f}%  "
            f"T_solo={s['temp_solo_C']:5.1f}°C  "
            f"UR_solo={s['umid_solo_pct']:5.1f}%  "
            f"| ETo={payload['penman_monteith']['eto_mm_30min']:.4f}mm  "
            f"| ⏱ {pred['horas_ate_irrigacao']:5.1f}h  "
            f"💧 {pred['volume_L_m2']:4.1f}L/m²  "
            f"🎯 {pred['confianca_pct']:4.1f}%  "
            f"{urgencia_str}"
        )
        sys.stdout.flush()

        time.sleep(intervalo)

    print("\n[INFO] Simulação concluída.", file=sys.stderr)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Publica leituras do gêmeo digital no Firebase em tempo real."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--real",
        action="store_true",
        help="Tempo real: 1 leitura a cada 30 minutos (produção).",
    )
    group.add_argument(
        "--rapido",
        type=float,
        default=2.0,
        metavar="N",
        help="Leituras por segundo (demo). Padrão: 2.0",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Imprime JSON no terminal sem conectar ao Firebase.",
    )
    args = parser.parse_args()

    modo = "real" if args.real else "rapido"
    executar(modo=modo, taxa=args.rapido, offline=args.offline)
