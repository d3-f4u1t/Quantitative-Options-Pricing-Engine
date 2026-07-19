#to RUN:
"""
    streamlit run QuantOS_Dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
import io

from option_Eng import MarketEnvironment, EuropeanCall, AsianCall, MonteCarloEngine, RiskManager, Portfolio

st.set_page_config(page_title="QuantOS Dashboard", layout="wide")
st.title("QuantOS: Pricing & Risk Engine")

# Side bar
st.sidebar.header("Engine Configuration")
mode = st.sidebar.radio("Select Strategy", ["Single Option Pricing", "Multi-Asset Portfolio Simulation"])
sims = st.sidebar.slider("Simulations", 1000, 50000, 10000, step=1000)
steps = st.sidebar.slider("Time Steps (Days)", 10, 252, 252)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.05, step=0.01)

# Helper functions
@st.cache_data
def get_index_tickers(index_name):
    """Scrapes Wikipedia using column index (safest method) to avoid header renaming errors."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        if index_name == "S&P 500":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))[0]
            # Grab the 0th column (Symbol) regardless of header name
            return df.iloc[:, 0].astype(str).str.replace('.', '-').tolist()
        elif index_name == "Nasdaq 100":
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            df = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))[4]
            # Grab the 0th column (Ticker) regardless of header name
            return df.iloc[:, 0].astype(str).str.replace('.', '-').tolist()
        return []
    except Exception as e:
        st.error(f"Could not fetch index constituents: {e}")
        return []

@st.cache_data
def fetch_data(tickers):
    # Sanitize inputs
    ticker_list = [t.strip().replace('.', '-') for t in tickers.split(',') if t.strip()]
    if not ticker_list: return None
    
    # Download data; multi_level_index=False flattens columns to just tickers
    data = yf.download(ticker_list, period="1y", multi_level_index=False)['Close']
    
    # Ensure it's a dataframe if only one ticker
    if isinstance(data, pd.Series):
        data = data.to_frame(name=ticker_list[0])
    
    # Clean data: drop any columns that are entirely empty (delisted/invalid tickers)
    data = data.dropna(axis=1, how='all')
    if data.empty: return None
    
    # Single Asset Logic
    if data.shape[1] == 1:
        spot = data.iloc[-1].values[0]
        vol = data.pct_change().dropna().iloc[:, 0].std() * np.sqrt(252)
        return spot, vol
    
    # Multi-Asset Logic
    returns = data.pct_change().dropna()
    spots = data.iloc[-1].values
    vols = returns.std().values * np.sqrt(252)
    corr_matrix = returns.corr().values
    return spots, vols, corr_matrix, data.columns.tolist()

# Logic
if mode == "Single Option Pricing":
    st.subheader("European & Asian Option Pricing Engine")
    ticker = st.text_input("Ticker Symbol", "AAPL")
    
    try:
        spot, vol = fetch_data(ticker)
        st.write(f"**Live Market Data:** Spot = ${spot:.2f} | Volatility = {vol*100:.2f}%")
        
        strike = st.number_input("Target Strike Price", value=float(spot * 1.05))
        ttm = st.number_input("Time to Maturity (Years)", value=1.0)
        
        if st.button("Run Simulation"):
            with st.spinner('Running Monte Carlo Engine...'):
                market = MarketEnvironment(spot, risk_free_rate, vol)
                euro_call = EuropeanCall(strike, ttm)
                engine = MonteCarloEngine(simulations=sims, steps=steps)
                price, paths = engine.price_option(market, euro_call, use_antithetic=True)
                
                risk_desk = RiskManager(engine)
                greeks = risk_desk.calculate_greeks(market, euro_call)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Fair Value (Price)", f"${price:.2f}")
                col2.metric("Delta (Stock Exposure)", f"{greeks['Delta']:.4f}")
                col3.metric("Vega (Vol Exposure)", f"{greeks['Vega']:.4f}")
                
                fig = go.Figure()
                num_paths_to_plot = min(100, sims)
                for i in range(num_paths_to_plot):
                    fig.add_trace(go.Scatter(y=paths[i, :], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                fig.add_hline(y=strike, line_dash="dash", line_color="red", annotation_text="Strike Price")
                fig.update_layout(title=f"Monte Carlo Universes ({num_paths_to_plot} plotted)", xaxis_title="Days", yaxis_title="Stock Price")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching data: {e}")

elif mode == "Multi-Asset Portfolio Simulation":
    st.subheader("Phase 3: Cholesky Correlation & Portfolio Risk")
    choice = st.selectbox("Market Index", ["Custom", "S&P 500", "Nasdaq 100"])
    
    if choice != "Custom":
        with st.spinner(f"Fetching {choice} constituents..."):
            ticker_list = get_index_tickers(choice)[:20]
            tickers = ",".join(ticker_list)
            st.write(f"Active Tickers: {tickers}")
    else:
        tickers = st.text_input("Enter Tickers (comma separated)", "AAPL,MSFT,NVDA,GOOGL")
    
    try:
        result = fetch_data(tickers)
        if result:
            spots, vols, corr_matrix, ticker_list = result
            st.write("### Calculated Asset Correlation Matrix")
            st.dataframe(pd.DataFrame(corr_matrix, columns=ticker_list, index=ticker_list).style.background_gradient(cmap='coolwarm'))
            
            if st.button("Run Correlated Portfolio"):
                with st.spinner('Entangling Assets using Cholesky Decomposition...'):
                    engine = MonteCarloEngine(simulations=sims, steps=steps)
                    rates = [risk_free_rate] * len(spots)
                    multi_paths = engine.generate_correlated_paths(spots, rates, vols, corr_matrix, ttm=1.0)
                    portfolio_paths = np.sum(multi_paths, axis=0)
                    
                    col1, col2 = st.columns(2)
                    initial_port_value = np.sum(spots)
                    var_95 = np.percentile(portfolio_paths[:, -1], 5)
                    col1.metric("Initial Portfolio Value", f"${initial_port_value:.2f}")
                    col2.metric("Worst Case Value (VaR 95%)", f"${var_95:.2f}", f"${var_95 - initial_port_value:.2f} potential loss")
                    
                    fig = go.Figure()
                    for i in range(min(100, sims)):
                        fig.add_trace(go.Scatter(y=portfolio_paths[i, :], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                    fig.update_layout(title="Correlated Portfolio Value Trajectories", xaxis_title="Days", yaxis_title="Total Portfolio Value ($)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not retrieve enough valid data for these tickers.")
    except Exception as e:
        st.error(f"Error in portfolio simulation: {e}")