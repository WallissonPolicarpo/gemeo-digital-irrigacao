"""
Gerador de dataset sintético para Gêmeo Digital de Irrigação
Gera dois arquivos:
  - dataset_historico.csv  (390 dias, 30min) → treinamento do modelo
  - dataset_futuro.csv     (14 dias, 30min)  → simulação "ao vivo"
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math

rng = np.random.default_rng(42)

# ─── Parâmetros físicos base ──────────────────────────────────────────────────
LATITUDE_RAD = math.radians(-16.7)  # Goiás (UFG/Goiânia)
ALTITUDE_M   = 748                  # altitude média Goiânia

def rad_extraterrestre(doy):
    """Radiação extraterrestre MJ/m²/dia — equação FAO 56"""
    dr  = 1 + 0.033 * math.cos(2*math.pi*doy/365)
    dec = 0.409 * math.sin(2*math.pi*doy/365 - 1.39)
    ws  = math.acos(-math.tan(LATITUDE_RAD)*math.tan(dec))
    Ra  = (24*60/math.pi) * 0.0820 * dr * (
        ws*math.sin(LATITUDE_RAD)*math.sin(dec) +
        math.cos(LATITUDE_RAD)*math.cos(dec)*math.sin(ws)
    )
    return Ra

def penman_monteith(Tmin, Tmax, Tmean, RH_mean, Ra, u2, P=101.325):
    """ETo diária simplificada (FAO 56, eq. 6) em mm/dia"""
    Tmean = float(Tmean)
    delta = 4098 * (0.6108 * math.exp(17.27*Tmean/(Tmean+237.3))) / (Tmean+237.3)**2
    gamma = 0.000665 * P
    es_Tmax = 0.6108 * math.exp(17.27*Tmax/(Tmax+237.3))
    es_Tmin = 0.6108 * math.exp(17.27*Tmin/(Tmin+237.3))
    es      = (es_Tmax + es_Tmin) / 2
    ea      = es * RH_mean / 100
    Rns     = (1-0.23) * Ra * 0.75 * 0.5   # simplificação de Rn
    ETo = (0.408*delta*Rns + gamma*(900/(Tmean+273))*u2*(es-ea)) / \
          (delta + gamma*(1+0.34*u2))
    return max(ETo, 0)

# ─── Geração por timestamp 30min ─────────────────────────────────────────────
def gerar_bloco(start_date, n_days, seed_offset=0):
    rng_local = np.random.default_rng(42 + seed_offset)
    timestamps = []
    rows = []

    base_time = datetime(start_date.year, start_date.month, start_date.day, 0, 0)
    n_steps   = n_days * 48  # 48 medições por dia (30min)

    # Estado do solo
    umid_solo_atual = 35.0  # % volumétrica inicial
    dias_ult_irrig  = 3
    acum_irrig_dia  = 0.0
    ultimo_dia      = -1

    # cache diário para ETo
    eto_diario_cache = {}

    for step in range(n_steps):
        ts    = base_time + timedelta(minutes=30*step)
        doy   = ts.timetuple().tm_yday
        hora  = ts.hour + ts.minute/60.0
        dia_n = step // 48

        # ── Temperatura do ar: ciclo sazonal + diurno + ruído ──────────────
        t_base  = 24 + 5*math.sin(2*math.pi*(doy-80)/365)   # sazonalidade
        t_diurn = 6  * math.sin(math.pi*(hora-6)/12) if 6<=hora<=18 else \
                 -3  * math.sin(math.pi*(hora-18)/12)
        t_ar    = t_base + t_diurn + rng_local.normal(0, 0.6)

        # ── Umidade relativa do ar ─────────────────────────────────────────
        rh_base  = 70 - 20*math.sin(2*math.pi*(doy-80)/365)
        rh_diurn = -15 * math.sin(math.pi*(hora-6)/12) if 6<=hora<=18 else \
                    8  * math.sin(math.pi*(hora-18)/12)
        rh_ar    = np.clip(rh_base + rh_diurn + rng_local.normal(0, 3), 20, 98)

        # ── Temperatura do solo ────────────────────────────────────────────
        t_solo  = t_ar - 3 + rng_local.normal(0, 0.4)

        # ── Radiação solar ─────────────────────────────────────────────────
        rad_max = max(0, 900 * math.sin(math.pi*(hora-6)/12)) if 6<=hora<=18 else 0
        nuvem   = np.clip(rng_local.beta(2,5), 0, 1) if 6<=hora<=18 else 0
        rad_sol = rad_max * (1 - 0.7*nuvem) + rng_local.normal(0, 5)
        rad_sol = max(0, rad_sol)

        # ── Vento ──────────────────────────────────────────────────────────
        vento = abs(rng_local.normal(2.5, 1.0))

        # ── ETo diária (cálculo único por dia) ────────────────────────────
        if doy not in eto_diario_cache:
            Ra   = rad_extraterrestre(doy)
            Tmin = t_base - 7 + rng_local.normal(0, 1)
            Tmax = t_base + 5 + rng_local.normal(0, 1)
            eto_diario_cache[doy] = penman_monteith(
                Tmin, Tmax, t_base, rh_base, Ra, 2.5
            )
        eto_d = eto_diario_cache[doy]
        eto_30min = eto_d / 48  # distribuir pelo dia

        # ── Umidade do solo ────────────────────────────────────────────────
        # Evapotranspiração reduz umidade; chuva/irrigação repõe
        kc    = 0.75   # coeficiente cultural médio
        etc   = eto_30min * kc
        umid_solo_atual -= etc * 0.5   # fator de extração simplificado
        umid_solo_atual  = max(umid_solo_atual, 10.0)

        # Chuva estocástica (mais provável nov-mar, cerrado goiano)
        prob_chuva = 0.003 if (doy < 90 or doy > 300) else 0.001
        if rng_local.random() < prob_chuva:
            chuva_vol = rng_local.uniform(5, 30)
            umid_solo_atual = min(umid_solo_atual + chuva_vol*0.4, 55.0)

        # ── Lógica de irrigação ────────────────────────────────────────────
        # Gatilho: umidade cai abaixo de limiar OU dias > threshold
        limiar_umid  = 22 + rng_local.normal(0, 2)
        limiar_dias  = 4 + rng_local.integers(0, 3)
        irrigou      = 0
        vol_irrig    = 0.0
        confianca    = 0.0

        if dia_n != ultimo_dia:
            dias_ult_irrig += 1
            acum_irrig_dia  = 0.0
            ultimo_dia      = dia_n

        if hora == 6.0 or hora == 6.5:  # avalia 1x por manhã
            deficit = max(0, 30 - umid_solo_atual)
            if umid_solo_atual < limiar_umid or dias_ult_irrig >= limiar_dias:
                irrigou   = 1
                # Volume baseado no déficit + ETo acumulada
                vol_irrig = deficit * 1.8 + eto_d * kc * 1.2
                vol_irrig = max(vol_irrig, 3.0)
                confianca = np.clip(
                    0.5 + (limiar_umid - umid_solo_atual)/50 +
                    (dias_ult_irrig/limiar_dias)*0.2, 0.3, 0.99
                )
                umid_solo_atual = min(umid_solo_atual + vol_irrig*0.35, 52.0)
                acum_irrig_dia += vol_irrig
                dias_ult_irrig  = 0

        umid_solo_step = np.clip(
            umid_solo_atual + rng_local.normal(0, 0.5), 10, 55
        )

        rows.append({
            'timestamp':              ts.strftime('%Y-%m-%d %H:%M:%S'),
            'temp_ar_C':              round(t_ar, 2),
            'umid_ar_pct':            round(rh_ar, 2),
            'temp_solo_C':            round(t_solo, 2),
            'umid_solo_pct':          round(umid_solo_step, 2),
            'radiacao_solar_Wm2':     round(rad_sol, 2),
            'vento_ms':               round(vento, 2),
            'eto_penman_monteith_mm': round(eto_30min, 4),
            'irrigou':                irrigou,
            'volume_irrigado_L_m2':   round(vol_irrig, 2),
            'dias_ultima_irrigacao':  dias_ult_irrig,
            'confianca_label':        round(confianca, 3),
        })

    return pd.DataFrame(rows)

print("Gerando dataset histórico (390 dias)...")
df_hist = gerar_bloco(datetime(2023, 7, 1), 390, seed_offset=0)
df_hist.to_csv('/home/claude/dataset_historico.csv', index=False)
print(f"  → {len(df_hist)} linhas | {df_hist['irrigou'].sum()} eventos de irrigação")

print("Gerando dataset futuro (14 dias)...")
df_fut = gerar_bloco(datetime(2024, 7, 31), 14, seed_offset=999)
df_fut.to_csv('/home/claude/dataset_futuro.csv', index=False)
print(f"  → {len(df_fut)} linhas | {df_fut['irrigou'].sum()} eventos de irrigação")

# ── Estatísticas rápidas ──────────────────────────────────────────────────────
print("\n── Estatísticas dataset histórico ──")
print(df_hist[['temp_ar_C','umid_ar_pct','temp_solo_C','umid_solo_pct',
               'eto_penman_monteith_mm','volume_irrigado_L_m2']].describe().round(2))
print(f"\nPrevalência irrigação: {df_hist['irrigou'].mean()*100:.2f}%")
print(f"Intervalo médio entre irrigações: "
      f"{df_hist[df_hist['irrigou']==1]['dias_ultima_irrigacao'].mean():.1f} dias")
