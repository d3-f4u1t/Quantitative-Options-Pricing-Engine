

#to RUN:
"""
    streamlit run QuantOS_Dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import time

from option_Eng import MarketEnvironment, EuropeanCall, AsianCall, MonteCarloEngine, RiskManager, Portfolio

st.set_page_config(page_title="QuantOS Dashboard", layout="wide")
st.title("QuantOS: Pricing & Risk Engine")

#Side bar
st.sidebar.header("Engine Configuration")
mode = st.sidebar.radio("Select Strategy", ["Single Option Pricing", "Multi-Asset Portfolio Simulation"])
sims = st.sidebar.slider("Simulations", 1000, 50000, 10000, step=1000)
steps = st.sidebar.slider("Time Steps (Days)", 10, 252, 252)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.05, step=0.01)

#helper function
@st.cache_data # Caches the data so we don't spam Yahoo Finance every time you move a slider
def fetch_data(tickers):
    ticker_list = [t.strip() for t in tickers.split(',')]
    if len(ticker_list) == 1:
        hist = yf.Ticker(ticker_list[0]).history(period="1y")
        spot = hist['Close'].iloc[-1]
        vol = hist['Close'].pct_change().dropna().std() * np.sqrt(252)
        return spot, vol
    else:
        data = yf.download(ticker_list, period="1y")['Close']
        returns = data.pct_change().dropna()
        spots = data.iloc[-1].values
        vols = returns.std().values * np.sqrt(252)
        corr_matrix = returns.corr().values
        return spots, vols, corr_matrix, ticker_list

#logic

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
                #Setup Backend Objects
                market = MarketEnvironment(spot, risk_free_rate, vol)
                euro_call = EuropeanCall(strike, ttm)
                engine = MonteCarloEngine(simulations=sims, steps=steps)
                
                #Backend Pricing
                price, paths = engine.price_option(market, euro_call, use_antithetic=True)
                
                #Backend Risk Management
                risk_desk = RiskManager(engine)
                greeks = risk_desk.calculate_greeks(market, euro_call)
                
                # Display Results
                col1, col2, col3 = st.columns(3)
                col1.metric("Fair Value (Price)", f"${price:.2f}")
                col2.metric("Delta (Stock Exposure)", f"{greeks['Delta']:.4f}")
                col3.metric("Vega (Vol Exposure)", f"{greeks['Vega']:.4f}")
                
                # Plotly Interactive Chart
                chart_placeholder = st.empty()
                num_paths_to_plot = min(100, sims) # Only plot 100 so browser doesn't crash
                y_min, y_max = np.min(paths[:num_paths_to_plot, :]), np.max(paths[:num_paths_to_plot, :])
                chunk_size = max(1, steps // 20)
                
                for frame in range(chunk_size, steps + chunk_size, chunk_size):
                    current_step = min(frame, steps)
                    fig = go.Figure()
                    for i in range(num_paths_to_plot):
                        fig.add_trace(go.Scatter(x=np.arange(current_step + 1), y=paths[i, :current_step + 1], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                    fig.add_hline(y=strike, line_dash="dash", line_color="red", annotation_text="Strike Price")
                    fig.update_layout(title=f"Monte Carlo Universes ({num_paths_to_plot} plotted) - Day {current_step}/{steps}", xaxis_title="Days", yaxis_title="Stock Price", xaxis=dict(range=[0, steps]), yaxis=dict(range=[y_min, y_max]))
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.05)
                
    except Exception as e:
        st.error(f"Error fetching data or running engine: {e}")

elif mode == "Multi-Asset Portfolio Simulation":
    st.subheader("Phase 3: Cholesky Correlation & Portfolio Risk")
    tickers = st.text_input("Enter Tickers (comma separated)", "AAPL,MSFT,NVDA,GOOGL")
    
    try:
        spots, vols, corr_matrix, ticker_list = fetch_data(tickers)
        rates = [risk_free_rate] * len(spots)
        
        # Display the Correlation Matrix
        st.write("### Calculated Asset Correlation Matrix")
        st.dataframe(pd.DataFrame(corr_matrix, columns=ticker_list, index=ticker_list).style.background_gradient(cmap='coolwarm'))
        
        if st.button("Run Correlated Portfolio"):
            with st.spinner('Entangling Assets using Cholesky Decomposition...'):
                #Setup Backend
                engine = MonteCarloEngine(simulations=sims, steps=steps)
                
                #Backend Multi-Asset Generator
                multi_paths = engine.generate_correlated_paths(spots, rates, vols, corr_matrix, ttm=1.0)
                
                # Calculate total portfolio value over time (Sum across all assets)
                # multi_paths shape is (num_assets, sims, steps)
                portfolio_paths = np.sum(multi_paths, axis=0) # shape becomes (sims, steps)
                
                # Display Results
                col1, col2 = st.columns(2)
                initial_port_value = np.sum(spots)
                expected_final_value = np.mean(portfolio_paths[:, -1])
                var_95 = np.percentile(portfolio_paths[:, -1], 5) # 5th percentile worst outcome
                
                col1.metric("Initial Portfolio Value", f"${initial_port_value:.2f}")
                col2.metric("Worst Case Value (VaR 95%)", f"${var_95:.2f}", f"${var_95 - initial_port_value:.2f} potential loss")
                
                # Plotly Chart of the PORTFOLIO value
                chart_placeholder = st.empty()
                num_paths_to_plot = min(100, sims)
                y_min, y_max = np.min(portfolio_paths[:num_paths_to_plot, :]), np.max(portfolio_paths[:num_paths_to_plot, :])
                chunk_size = max(1, steps // 20)
                
                for frame in range(chunk_size, steps + chunk_size, chunk_size):
                    current_step = min(frame, steps)
                    fig = go.Figure()
                    for i in range(num_paths_to_plot):
                        fig.add_trace(go.Scatter(x=np.arange(current_step + 1), y=portfolio_paths[i, :current_step + 1], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
                    fig.update_layout(title=f"Correlated Portfolio Value Trajectories - Day {current_step}/{steps}", xaxis_title="Days", yaxis_title="Total Portfolio Value ($)", xaxis=dict(range=[0, steps]), yaxis=dict(range=[y_min, y_max]))
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.05)

    except Exception as e:
        st.error(f"Need at least 2 valid tickers for correlation. Error: {e}")