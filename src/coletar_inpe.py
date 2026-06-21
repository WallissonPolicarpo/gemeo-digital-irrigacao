"""
Coleta de dados meteorológicos do INPE/CPTEC
Fontes:
  - CPTEC/INPE API de previsão numérica (API pública)
  - INMET (Instituto Nacional de Meteorologia) — fallback
  - Open-Meteo (fallback gratuito, sem chave)

Este script tenta as fontes em ordem e salva o resultado.
"""
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import time

# Coordenadas da área de cultivo (exemplo: Goiânia)
LAT   = -16.6864
LON   = -49.2643
CIDADE_CPTEC = 244   # código Goiânia no CPTEC

# ── Fonte 1: CPTEC/INPE ────────────────────────────────────────────────────────
def coletar_cptec(cidade_id: int) -> dict | None:
    """
    API pública CPTEC: http://servicos.cptec.inpe.br/XML/cidade/7dias/{id}/previsao.xml
    Retorna previsão de 7 dias em XML
    """
    try:
        url = f"http://servicos.cptec.inpe.br/XML/cidade/7dias/{cidade_id}/previsao.xml"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        # Parse XML simples
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        previsoes = []
        for p in root.findall('.//previsao'):
            previsoes.append({
                'data':        p.findtext('dia', ''),
                'tempo':       p.findtext('tempo', ''),
                'maxima_C':    float(p.findtext('maxima', '0') or 0),
                'minima_C':    float(p.findtext('minima', '0') or 0),
                'iuv':         float(p.findtext('iuv', '0') or 0),
                'fonte':       'CPTEC/INPE'
            })
        return {'status': 'ok', 'dados': previsoes}
    except Exception as e:
        return {'status': 'erro', 'msg': str(e)}

# ── Fonte 2: Open-Meteo (fallback gratuito, sem chave de API) ─────────────────
def coletar_open_meteo(lat: float, lon: float, dias: int = 7) -> dict | None:
    """
    Open-Meteo: API aberta, alinhada com padrões WMO
    Retorna temperatura, umidade, precipitação, vento — horário ou diário
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude':               lat,
            'longitude':              lon,
            'hourly':                 ','.join([
                'temperature_2m',
                'relativehumidity_2m',
                'windspeed_10m',
                'precipitation',
                'shortwave_radiation',
                'et0_fao_evapotranspiration',   # ETo pronta!
            ]),
            'daily':                  ','.join([
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'et0_fao_evapotranspiration',
                'windspeed_10m_max',
            ]),
            'timezone':               'America/Sao_Paulo',
            'forecast_days':          dias,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Constrói DataFrame horário
        hourly = data.get('hourly', {})
        df_h = pd.DataFrame({
            'timestamp':    hourly.get('time', []),
            'temp_ar_C':    hourly.get('temperature_2m', []),
            'umid_ar_pct':  hourly.get('relativehumidity_2m', []),
            'vento_ms':     hourly.get('windspeed_10m', []),
            'precipitacao': hourly.get('precipitation', []),
            'radiacao_Wm2': hourly.get('shortwave_radiation', []),
            'eto_inpe_mm':  hourly.get('et0_fao_evapotranspiration', []),
            'fonte':        'Open-Meteo'
        })

        # Constrói DataFrame diário
        daily = data.get('daily', {})
        df_d = pd.DataFrame({
            'data':        daily.get('time', []),
            'temp_max_C':  daily.get('temperature_2m_max', []),
            'temp_min_C':  daily.get('temperature_2m_min', []),
            'precip_mm':   daily.get('precipitation_sum', []),
            'eto_d_mm':    daily.get('et0_fao_evapotranspiration', []),
            'vento_max':   daily.get('windspeed_10m_max', []),
            'fonte':       'Open-Meteo'
        })

        return {'status': 'ok', 'horario': df_h, 'diario': df_d}
    except Exception as e:
        return {'status': 'erro', 'msg': str(e)}

# ── Orquestrador ──────────────────────────────────────────────────────────────
def coletar_previsao_inpe(lat=LAT, lon=LON, cidade_id=CIDADE_CPTEC):
    print(f"[{datetime.now():%H:%M:%S}] Tentando CPTEC/INPE...")
    resultado_cptec = coletar_cptec(cidade_id)
    if resultado_cptec['status'] == 'ok':
        print(f"  ✓ CPTEC/INPE: {len(resultado_cptec['dados'])} dias de previsão")

    print(f"[{datetime.now():%H:%M:%S}] Tentando Open-Meteo (fallback)...")
    resultado_om = coletar_open_meteo(lat, lon, dias=7)
    if resultado_om['status'] == 'ok':
        print(f"  ✓ Open-Meteo: {len(resultado_om['horario'])} registros horários")
        resultado_om['horario'].to_csv('/home/claude/previsao_horaria.csv', index=False)
        resultado_om['diario'].to_csv('/home/claude/previsao_diaria.csv', index=False)
        print("  Arquivos salvos: previsao_horaria.csv, previsao_diaria.csv")
    else:
        print(f"  ✗ Open-Meteo erro: {resultado_om['msg']}")

    return resultado_cptec, resultado_om

if __name__ == '__main__':
    coletar_previsao_inpe()
