"""
Simulador de leitura 'ao vivo' — para demonstração sem sensores físicos
Lê o dataset_futuro_com_predicoes.csv linha por linha
e emite eventos como se fossem chegando em tempo real (ou comprimidos).

Modos:
  --real      : espera 30 minutos reais entre leituras (uso em produção)
  --rapido N  : emite N leituras por segundo (uso em demo/apresentação)
  --json      : saída como JSON (para consumo pelo backend/dashboard)
"""
import time, sys, json, argparse
import pandas as pd
from datetime import datetime

def formatar_alerta(row):
    h = float(row['pred_horas_ate_irrigacao'])
    v = float(row['pred_volume_L_m2'])
    c = float(row['pred_confianca_pct'])

    if h <= 4:
        urgencia = '🔴 URGENTE'
    elif h <= 12:
        urgencia = '🟡 ATENÇÃO'
    elif h <= 24:
        urgencia = '🟢 PROGRAMAR'
    else:
        urgencia = '✅ OK'

    return {
        'timestamp':            str(row['timestamp']),
        'sensores': {
            'temp_ar_C':        round(float(row['temp_ar_C']), 1),
            'umid_ar_pct':      round(float(row['umid_ar_pct']), 1),
            'temp_solo_C':      round(float(row['temp_solo_C']), 1),
            'umid_solo_pct':    round(float(row['umid_solo_pct']), 1),
        },
        'eto_mm_30min':         round(float(row['eto_penman_monteith_mm']), 4),
        'predicao': {
            'horas_ate_irrigacao': round(h, 1),
            'volume_L_m2':         round(v, 2),
            'confianca_pct':       round(c, 1),
            'urgencia':            urgencia,
        }
    }

def simular(modo='rapido', taxa=2.0, saida_json=False):
    df = pd.read_csv('/home/claude/dataset_futuro_com_predicoes.csv')
    print(f"[Simulador] {len(df)} registros carregados. Modo: {modo}", file=sys.stderr)
    print(f"[Simulador] Iniciando em 2s...", file=sys.stderr)
    time.sleep(2)

    for _, row in df.iterrows():
        evento = formatar_alerta(row)
        if saida_json:
            print(json.dumps(evento, ensure_ascii=False))
        else:
            ts      = evento['timestamp']
            s       = evento['sensores']
            pred    = evento['predicao']
            print(f"[{ts}] "
                  f"T_ar={s['temp_ar_C']:5.1f}°C  UR={s['umid_ar_pct']:5.1f}%  "
                  f"T_solo={s['temp_solo_C']:5.1f}°C  UR_solo={s['umid_solo_pct']:5.1f}%  "
                  f"| ETo={evento['eto_mm_30min']:.4f}mm  "
                  f"| ⏱ {pred['horas_ate_irrigacao']:5.1f}h  "
                  f"💧 {pred['volume_L_m2']:5.2f}L/m²  "
                  f"🎯 {pred['confianca_pct']:5.1f}%  "
                  f"{pred['urgencia']}")
        sys.stdout.flush()

        if modo == 'real':
            time.sleep(1800)
        else:
            time.sleep(1.0 / taxa)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real',   action='store_true')
    parser.add_argument('--rapido', type=float, default=4.0,
                        help='leituras por segundo')
    parser.add_argument('--json',   action='store_true')
    args = parser.parse_args()

    modo = 'real' if args.real else 'rapido'
    simular(modo=modo, taxa=args.rapido, saida_json=args.json)
