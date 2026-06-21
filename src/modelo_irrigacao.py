"""
Pipeline de treinamento e inferência — Gêmeo Digital de Irrigação
Modelo: XGBoost com regressão multitarefa (via MultiOutputRegressor)
Saídas: quando_irrigar_horas, volume_L_m2, confianca_pct
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import warnings, joblib
warnings.filterwarnings('ignore')

# ── 1. Carrega e prepara features ──────────────────────────────────────────────
print("Carregando dataset histórico...")
df = pd.read_csv('/home/claude/dataset_historico.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Features de janela deslizante (últimas 6h = 12 leituras)
WINDOW = 12

for col in ['temp_ar_C', 'umid_ar_pct', 'temp_solo_C', 'umid_solo_pct',
            'radiacao_solar_Wm2', 'vento_ms', 'eto_penman_monteith_mm']:
    df[f'{col}_mean6h']  = df[col].rolling(WINDOW, min_periods=1).mean()
    df[f'{col}_std6h']   = df[col].rolling(WINDOW, min_periods=1).std().fillna(0)
    df[f'{col}_min6h']   = df[col].rolling(WINDOW, min_periods=1).min()
    df[f'{col}_max6h']   = df[col].rolling(WINDOW, min_periods=1).max()

# Features temporais
df['hora']            = df['timestamp'].dt.hour
df['mes']             = df['timestamp'].dt.month
df['doy']             = df['timestamp'].dt.dayofyear
df['hora_sin']        = np.sin(2*np.pi*df['hora']/24)
df['hora_cos']        = np.cos(2*np.pi*df['hora']/24)
df['doy_sin']         = np.sin(2*np.pi*df['doy']/365)
df['doy_cos']         = np.cos(2*np.pi*df['doy']/365)

# ETo acumulada últimas 24h → déficit hídrico estimado
df['eto_acum_24h']    = df['eto_penman_monteith_mm'].rolling(48, min_periods=1).sum()

# Engenharia de features para o label de tempo até próxima irrigação
# Calcula horas até o próximo evento de irrigação
eventos = df[df['irrigou'] == 1].index.tolist()
df['horas_ate_irrigacao'] = np.nan
for i in range(len(df)):
    proximos = [e for e in eventos if e >= i]
    if proximos:
        df.loc[i, 'horas_ate_irrigacao'] = (proximos[0] - i) * 0.5  # 30min por step
df['horas_ate_irrigacao'] = df['horas_ate_irrigacao'].fillna(72)
df['horas_ate_irrigacao'] = df['horas_ate_irrigacao'].clip(0, 72)

# Target: volume médio previsto (próximo evento)
df['volume_previsto'] = 0.0
for idx in eventos:
    vol = df.loc[idx, 'volume_irrigado_L_m2']
    # Atribui o volume do próximo evento para as 24h anteriores
    inicio = max(0, idx - 48)
    df.loc[inicio:idx, 'volume_previsto'] = vol
df['volume_previsto'] = df['volume_previsto'].fillna(df['volume_irrigado_L_m2'].mean())

# ── 2. Seleciona features finais ───────────────────────────────────────────────
FEATURE_COLS = [
    # Estado atual dos sensores
    'temp_ar_C', 'umid_ar_pct', 'temp_solo_C', 'umid_solo_pct',
    'radiacao_solar_Wm2', 'vento_ms', 'eto_penman_monteith_mm',
    # Janela histórica 6h
    'temp_ar_C_mean6h', 'umid_ar_pct_mean6h', 'temp_solo_C_mean6h',
    'umid_solo_pct_mean6h', 'umid_solo_pct_std6h', 'umid_solo_pct_min6h',
    'eto_penman_monteith_mm_mean6h',
    # Temporal
    'hora_sin', 'hora_cos', 'doy_sin', 'doy_cos',
    # Déficit acumulado
    'eto_acum_24h', 'dias_ultima_irrigacao',
]

TARGET_COLS = ['horas_ate_irrigacao', 'volume_previsto', 'confianca_label']

df_clean = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
X = df_clean[FEATURE_COLS].values
y = df_clean[TARGET_COLS].values

print(f"Shape X: {X.shape}, Shape y: {y.shape}")

# ── 3. Split treino/validação (70/30, sem shuffle — respeita temporalidade) ───
split = int(0.70 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

# ── 4. Treinamento ─────────────────────────────────────────────────────────────
print("Treinando XGBoost MultiOutput...")
base_xgb = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model = MultiOutputRegressor(base_xgb)
model.fit(X_train_s, y_train)

# ── 5. Avaliação ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_val_s)
y_pred[:, 0] = np.clip(y_pred[:, 0], 0, 72)   # horas: 0–72
y_pred[:, 1] = np.clip(y_pred[:, 1], 0, 50)   # volume: 0–50 L/m²
y_pred[:, 2] = np.clip(y_pred[:, 2], 0, 1)    # confiança: 0–1

labels = ['horas_ate_irrigacao', 'volume_L_m2', 'confianca']
print("\n── Métricas de validação ──")
for i, lbl in enumerate(labels):
    mae = mean_absolute_error(y_val[:, i], y_pred[:, i])
    r2  = r2_score(y_val[:, i], y_pred[:, i])
    print(f"  {lbl:25s}  MAE={mae:.3f}  R²={r2:.3f}")

# ── 6. Importância das features ────────────────────────────────────────────────
print("\n── Top-10 features mais importantes (target: horas_ate_irrigacao) ──")
importances = model.estimators_[0].feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:10]
for feat, imp in feat_imp:
    bar = '█' * int(imp*200)
    print(f"  {feat:35s} {imp:.4f} {bar}")

# ── 7. Salva artefatos ────────────────────────────────────────────────────────
joblib.dump(model, '/home/claude/modelo_irrigacao.pkl')
joblib.dump(scaler, '/home/claude/scaler_irrigacao.pkl')
joblib.dump(FEATURE_COLS, '/home/claude/feature_cols.pkl')
print("\nModelo e scaler salvos com sucesso.")

# ── 8. Demonstração de inferência com dataset futuro ──────────────────────────
print("\n── Inferência no dataset futuro (simulação ao vivo) ──")
df_fut = pd.read_csv('/home/claude/dataset_futuro.csv', parse_dates=['timestamp'])
df_fut = df_fut.sort_values('timestamp').reset_index(drop=True)

# Mesmo pipeline de features (simplificado sem janela, pois é curto)
for col in ['temp_ar_C', 'umid_ar_pct', 'temp_solo_C', 'umid_solo_pct',
            'radiacao_solar_Wm2', 'vento_ms', 'eto_penman_monteith_mm']:
    df_fut[f'{col}_mean6h']  = df_fut[col].rolling(WINDOW, min_periods=1).mean()
    df_fut[f'{col}_std6h']   = df_fut[col].rolling(WINDOW, min_periods=1).std().fillna(0)
    df_fut[f'{col}_min6h']   = df_fut[col].rolling(WINDOW, min_periods=1).min()
    df_fut[f'{col}_max6h']   = df_fut[col].rolling(WINDOW, min_periods=1).max()

df_fut['hora']          = df_fut['timestamp'].dt.hour
df_fut['mes']           = df_fut['timestamp'].dt.month
df_fut['doy']           = df_fut['timestamp'].dt.dayofyear
df_fut['hora_sin']      = np.sin(2*np.pi*df_fut['hora']/24)
df_fut['hora_cos']      = np.cos(2*np.pi*df_fut['hora']/24)
df_fut['doy_sin']       = np.sin(2*np.pi*df_fut['doy']/365)
df_fut['doy_cos']       = np.cos(2*np.pi*df_fut['doy']/365)
df_fut['eto_acum_24h']  = df_fut['eto_penman_monteith_mm'].rolling(48, min_periods=1).sum()
df_fut['dias_ultima_irrigacao'] = df_fut['dias_ultima_irrigacao']

X_fut = df_fut[FEATURE_COLS].fillna(0).values
X_fut_s = scaler.transform(X_fut)
preds = model.predict(X_fut_s)

preds[:, 0] = np.clip(preds[:, 0], 0, 72)
preds[:, 1] = np.clip(preds[:, 1], 0, 50)
preds[:, 2] = np.clip(preds[:, 2], 0, 1)

df_fut['pred_horas_ate_irrigacao'] = preds[:, 0].round(1)
df_fut['pred_volume_L_m2']         = preds[:, 1].round(2)
df_fut['pred_confianca_pct']       = (preds[:, 2] * 100).round(1)

# Exibe alertas previstos (próximas 24h)
alertas = df_fut[df_fut['pred_horas_ate_irrigacao'] <= 24].head(10)
if len(alertas):
    print(f"  Alertas de irrigação prevista nas próximas 24h:")
    print(alertas[['timestamp','pred_horas_ate_irrigacao',
                   'pred_volume_L_m2','pred_confianca_pct']].to_string(index=False))
else:
    print("  Nenhum alerta nas próximas 24h do dataset futuro.")

df_fut.to_csv('/home/claude/dataset_futuro_com_predicoes.csv', index=False)
print("\nDataset futuro com predições salvo.")
