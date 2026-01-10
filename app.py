import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Backtesting Pro - Momentum", layout="wide")
st.title("🧪 Backtesting: Estrategia de Momentum Configurable")

NOMBRES_IBEX = {
    "ACS.MC": "ACS", "ACX.MC": "Acerinox", "AENA.MC": "Aena", "AMS.MC": "Amadeus",
    "ANA.MC": "Acciona", "ANE.MC": "Acciona Energía", "BBVA.MC": "BBVA", "BKT.MC": "Bankinter",
    "CABK.MC": "CaixaBank", "CLNX.MC": "Cellnex", "COL.MC": "Colonial", "ELE.MC": "Endesa",
    "ENG.MC": "Enagás", "FDR.MC": "Fluidra", "FER.MC": "Ferrovial", "GRF.MC": "Grifols",
    "IAG.MC": "IAG (Iberia)", "IBE.MC": "Iberdrola", "IDR.MC": "Indra", "ITX.MC": "Inditex",
    "LOG.MC": "Logista", "MAP.MC": "Mapfre", "MRL.MC": "Merlin Prop.", "MTS.MC": "ArcelorMittal",
    "NTGY.MC": "Naturgy", "PUIG.MC": "Puig Brands", "RED.MC": "Redeia", "REP.MC": "Repsol",
    "ROVI.MC": "Rovi", "SAB.MC": "Sabadell", "SAN.MC": "Santander", "SCYR.MC": "Sacyr",
    "SLR.MC": "Solaria", "TEF.MC": "Telefónica", "UNI.MC": "Unicaja"
}

# --- FUNCIONES TÉCNICAS ---

def calcular_rsi(series, period=21):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_score_tecnico(row, p):
    score = 0
    price = row['Close']
    sma50 = row['SMA50']
    sma200 = row['SMA200']
    rsi = row['RSI']
    macd = row['MACD']
    signal = row['Signal']
    rvol = row['RVol']
    volat = row['Volat_Anual']
    dist_sma50 = row['Dist_SMA50']

    # 1. TENDENCIA
    if price > sma50 and price > sma200:
        score += p['w_sma_bull_cross'] if sma50 > sma200 else p['w_sma_bull_simple']
    elif price > sma200:
        score += p['w_sma_min']

    # 2. SOBREEXTENSIÓN
    if dist_sma50 > p['thr_overextended']: 
        score -= p['pen_overextended']
    elif abs(dist_sma50) < p['thr_near_sma']: 
        score += p['w_near_sma']

    # 3. MOMENTUM RSI
    if p['rsi_high_min'] <= rsi <= p['rsi_high_max']: 
        score += p['w_rsi_hot']
    elif p['rsi_mid_min'] <= rsi < p['rsi_high_min']: 
        score += p['w_rsi_mid']
    
    if rsi > p['rsi_overbought']: score -= p['pen_rsi_ob']
    if rsi < p['rsi_oversold']: score += p['w_rsi_os']

    # 4. MACD
    if macd > signal and macd > 0: score += p['w_macd_strong']
    elif macd > signal: score += p['w_macd_cross']

    # 5. VOLUMEN
    if rvol > p['thr_vol_high']: score += p['w_vol_high']
    elif rvol > p['thr_vol_mid']: score += p['w_vol_mid']

    # 6. VOLATILIDAD
    if volat > p['thr_volat_high']: score -= p['pen_volat_high']
    elif volat > p['thr_volat_mid']: score -= p['pen_volat_mid']

    return max(0, min(100, score))

# --- MOTOR DE BACKTEST ---

def run_backtest(ticker, start_date, end_date, capital_inicial, threshold_buy, threshold_sell, params):
    start_dt = datetime.combine(start_date, datetime.min.time()) - timedelta(days=365)
    df = yf.download(ticker, start=start_dt, end=end_date, progress=False)
    
    if df.empty: return None, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['RSI'] = calcular_rsi(df['Close'], 21)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RVol'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Volat_Anual'] = df['Close'].pct_change().rolling(20).std() * (252**0.5) * 100
    df['Dist_SMA50'] = ((df['Close'] - df['SMA50']) / df['SMA50']) * 100
    
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df = df.loc[start_date:]
    df['Score'] = df.apply(lambda row: calcular_score_tecnico(row, params), axis=1)

    posicion, capital, shares, stop_loss = 0, capital_inicial, 0, 0
    historial_trades, equity_curve = [], []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_score = df['Score'].iloc[i]
        current_date = df.index[i]
        current_atr = df['ATR'].iloc[i]

        if posicion == 1:
            if current_score < threshold_sell or current_price <= stop_loss:
                capital = shares * current_price
                motivo = "Stop Loss" if current_price <= stop_loss else "Bajo Score"
                historial_trades.append({"Fecha Venta": current_date, "Precio Venta": current_price, "Motivo": motivo, "Capital": capital})
                shares, posicion = 0, 0
        elif posicion == 0:
            if current_score >= threshold_buy:
                shares = capital / current_price
                stop_loss = current_price - (2 * current_atr)
                posicion = 1
                historial_trades.append({"Fecha Compra": current_date, "Precio Compra": current_price, "Score Entrada": current_score})

        equity_curve.append(shares * current_price if posicion == 1 else capital)

    df['Equity'] = equity_curve
    return df, historial_trades

# --- INTERFAZ ---

with st.sidebar:
    st.header("⚙️ Configuración General")
    ticker_sel = st.selectbox("Empresa", list(NOMBRES_IBEX.keys()))
    fecha_inicio = st.date_input("Inicio", datetime(2023, 1, 1))
    fecha_fin = st.date_input("Fin", datetime.now())
    cap_inicial = st.number_input("Capital (€)", value=10000)
    
    buy_score = st.slider("Score Compra (Gatillo)", 40, 90, 60)
    sell_score = st.slider("Score Venta (Salida)", 20, 60, 40)

    st.header("🛠️ Pesos del Score Técnico")
    with st.expander("📈 Tendencia"):
        p_sma_bull_cross = st.slider("Precio > SMA50 > SMA200", 0, 50, 50)
        p_sma_bull_simple = st.slider("Precio > SMA50 y SMA200", 0, 50, 50)
        p_sma_min = st.slider("Precio > SMA200 solamente", 0, 30, 30)
        
    with st.expander("🚀 Momentum (RSI 21)"):
        rsi_range = st.slider("Rango Ideal (Puntos Max)", 0, 100, (60, 75))
        p_rsi_hot = st.slider("Puntos en Rango Ideal", 0, 40, 25)
        p_rsi_ob = st.slider("Penalización Sobrecompra (>75)", 0, 30, 8)
        
    with st.expander("📊 MACD y Volumen"):
        p_macd_strong = st.slider("MACD > Signal & > 0", 0, 20, 10)
        p_vol_high = st.slider("Volumen > 1.5x media", 0, 15, 6)
        
    with st.expander("⚠️ Riesgo y Volatilidad"):
        p_dist_max = st.slider("Distancia a SMA50 max (%)", 5, 30, 15)
        p_pen_dist = st.slider("Penalización Sobreextensión", 0, 30, 10)
        p_pen_volat = st.slider("Penalización Volatilidad Alta", 0, 40, 20)

    params = {
        'w_sma_bull_cross': p_sma_bull_cross, 'w_sma_bull_simple': p_sma_bull_simple, 'w_sma_min': p_sma_min,
        'thr_overextended': p_dist_max, 'pen_overextended': p_pen_dist, 'thr_near_sma': 5, 'w_near_sma': 5,
        'rsi_high_min': rsi_range[0], 'rsi_high_max': rsi_range[1], 'w_rsi_hot': p_rsi_hot,
        'rsi_mid_min': 55, 'w_rsi_mid': 18, 'rsi_overbought': 75, 'pen_rsi_ob': p_rsi_ob,
        'rsi_oversold': 30, 'w_rsi_os': 5, 'w_macd_strong': p_macd_strong, 'w_macd_cross': 5,
        'thr_vol_high': 1.5, 'w_vol_high': p_vol_high, 'thr_vol_mid': 1.1, 'w_vol_mid': 3,
        'thr_volat_high': 35, 'pen_volat_high': p_pen_volat, 'thr_volat_mid': 30, 'pen_volat_mid': 10
    }

if st.sidebar.button("🚀 Ejecutar Backtest"):
    df_res, trades = run_backtest(ticker_sel, fecha_inicio, fecha_fin, cap_inicial, buy_score, sell_score, params)

    if df_res is not None:
        # --- CÁLCULO DE MÉTRICAS ---
        retorno_total = ((df_res['Equity'].iloc[-1] - cap_inicial) / cap_inicial) * 100
        buy_hold = ((df_res['Close'].iloc[-1] - df_res['Close'].iloc[0]) / df_res['Close'].iloc[0]) * 100
        
        # Funciones de Riesgo
        def calcular_max_drawdown(equity_series):
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            return drawdown.min() * 100

        def calcular_sharpe(equity_series):
            returns = equity_series.pct_change().dropna()
            if returns.std() == 0: return 0
            return (returns.mean() / returns.std()) * np.sqrt(252)

        m_drawdown = calcular_max_drawdown(df_res['Equity'])
        m_sharpe = calcular_sharpe(df_res['Equity'])

        # Procesamiento de trades para Win Rate
        clean_trades = []
        if trades:
            for i in range(0, len(trades), 2):
                if i+1 < len(trades):
                    t_in, t_out = trades[i], trades[i+1]
                    beneficio = ((t_out['Precio Venta'] - t_in['Precio Compra']) / t_in['Precio Compra']) * 100
                    clean_trades.append({
                        "Entrada": t_in['Fecha Compra'].date(), "Precio Ent.": round(t_in['Precio Compra'], 2),
                        "Salida": t_out['Fecha Venta'].date(), "Precio Sal.": round(t_out['Precio Venta'], 2),
                        "Motivo": t_out['Motivo'], "% Trade": beneficio
                    })
        
        win_rate = (len([t for t in clean_trades if t['% Trade'] > 0]) / len(clean_trades) * 100) if clean_trades else 0

        # --- PANEL DE RESULTADOS ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Estrategia", f"{retorno_total:.2f}%", f"{retorno_total - buy_hold:.2f}% vs B&H")
        c2.metric("Buy & Hold", f"{buy_hold:.2f}%")
        c3.metric("Capital Final", f"{df_res['Equity'].iloc[-1]:.2f} €")

        st.subheader("📊 Análisis de Riesgo y Eficiencia")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Drawdown", f"{m_drawdown:.2f}%")
        m2.metric("Ratio Sharpe", f"{m_sharpe:.2f}")
        m3.metric("Win Rate", f"{win_rate:.1f}%")
        m4.metric("Nº Operaciones", len(clean_trades))

        # --- GRÁFICO DE EQUITY ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Equity'], name='Equity (Estrategia)'))
        price_norm = df_res['Close'] * (cap_inicial / df_res['Close'].iloc[0])
        fig.add_trace(go.Scatter(x=df_res.index, y=price_norm, name='Precio (Norm)', line=dict(dash='dot', color='rgba(0,176,246,0.5)')))
        fig.update_layout(title="Curva de Equidad vs Activo Normalizado", xaxis_title="Fecha", yaxis_title="Capital (€)")
        st.plotly_chart(fig, use_container_width=True)

        # --- TABLA DE TRADES ---
        if clean_trades:
            st.subheader("📜 Historial de Operaciones")
            df_tabla = pd.DataFrame(clean_trades)
            # Formatear el % para visualización
            df_tabla['% Trade'] = df_tabla['% Trade'].map("{:.2f}%".format)
            st.table(df_tabla)

        # --- GRÁFICO DE SCORE ---
        st.subheader("📈 Evolución del Score Técnico")
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(x=df_res.index, y=df_res['Score'], name='Score', line=dict(color='orange')))
        fig_score.add_hline(y=buy_score, line_dash="dash", line_color="green", annotation_text="Gatillo Compra")
        fig_score.add_hline(y=sell_score, line_dash="dash", line_color="red", annotation_text="Gatillo Venta")
        st.plotly_chart(fig_score, use_container_width=True)
