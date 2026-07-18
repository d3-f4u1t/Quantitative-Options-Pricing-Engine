import numpy as np
import time
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import yfinance as yf
from matplotlib.animation import FuncAnimation

#Function to fetch real data
def get_market_data(ticker_symbol):
    """Fetches real price and calc volatility from Yahoo Finance."""
    print(f"Fetching live data for {ticker_symbol}...")
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")
    spot = hist['Close'].iloc[-1]

    # Calculate historical volatility (daily std dev * sqrt(252))
    returns = hist['Close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    return spot, vol

class MarketEnvironment:
    """
    Holds the state of the real world. 
    In a real firm, this data would be streamed live from Bloomberg or Reuters.
    """
    def __init__(self, spot_price, risk_free_rate, volatility):
        self.spot = spot_price          # CURRENT PRICE
        self.rate = risk_free_rate      # BASELINE INTEREST RATE
        self.vol = volatility           # HOW MUCH UP AND DOWN DOES A STOCK DO

class OptionContract:
    """Base class for all options. Every option needs a strike and a time to maturity."""
    def __init__(self, strike_price, time_to_maturity):
        self.strike = strike_price      # TARGET PRICE
        self.ttm = time_to_maturity     # WE ARE SIMULATING X YEARS INTO THE FUTURE

    def calculate_payoff(self, price_paths):
        """Every specific option type will override this method with its own rules."""
        raise NotImplementedError("Payoff must be defined in the subclass.")

class EuropeanCall(OptionContract):
    """European Options only care about the very last day."""
    
    def calculate_payoff(self, price_paths):
        # final price
        final_prices = price_paths[:, -1]
        # cal payoff- is strick price beat if not then 0 payoff
        payoffs = np.maximum(final_prices - self.strike, 0)
        return payoffs

    def black_scholes_anly(self, market: MarketEnvironment):
        # d1 and d2 are complex probability boundaries in the formula
        d1 = (math.log(market.spot / self.strike) + (market.rate + 0.5 * market.vol ** 2) * self.ttm) / (market.vol * math.sqrt(self.ttm))
        d2 = d1 - market.vol * math.sqrt(self.ttm)

        # cal the final theo price
        call_price = market.spot * stats.norm.cdf(d1) - self.strike * math.exp(-market.rate * self.ttm) * stats.norm.cdf(d2)
        return call_price

class AsianCall(OptionContract):
    """Asian Options care about the AVERAGE price over the whole timeline."""
    
    def calculate_payoff(self, price_paths):
        # avg horizontally across all days
        average_prices = np.mean(price_paths, axis=1)
        # cal payoff
        payoffs = np.maximum(average_prices - self.strike, 0)
        return payoffs

class MonteCarloEngine:
    """
    The mathematical factory. It takes a Market and an Option, 
    generates parallel universes, and returns the fair price.
    """
    def __init__(self, simulations, steps):
        self.sims = simulations
        self.steps = steps

    def generate_paths(self, market: MarketEnvironment, ttm, use_antithetic=True):
        # dt is time step
        dt = ttm / self.steps
        
        if use_antithetic:
            half_sims = self.sims // 2
            z_half = np.random.standard_normal((half_sims, self.steps))
            # taking out the noice and stacking it on to its neg value
            z_full = np.concatenate((z_half, -z_half), axis=0)
            z = z_full
        else:
            # Grid this will gen random stocks
            z = np.random.standard_normal((self.sims, self.steps))

        # cal daily growth for the entire grid
        growth_factors = np.exp((market.rate - 0.5 * market.vol ** 2) * dt + market.vol * math.sqrt(dt) * z)
        
        # timeline for linking the days together
        # axices = 1 multiplies left to right across column
        price_paths = market.spot * np.cumprod(growth_factors, axis=1)
        return price_paths
    
    def price_option(self, market: MarketEnvironment, option: OptionContract, use_antithetic=True):
        """Prices ANY option contract passed into it."""
        #Generate dif factors/ universe
        paths = self.generate_paths(market, option.ttm, use_antithetic)
        
        #specific option calculate own payoffs
        payoffs = option.calculate_payoff(paths)
        
        # Average and discount
        average_payoff = np.mean(payoffs)
        discounted_payoff = math.exp(-market.rate * option.ttm) * average_payoff
        
        return discounted_payoff, paths

    def generate_heston_paths(self, market: MarketEnvironment, ttm, kappa, theta, xi, rho):
        """
        PHASE 2: THE HESTON MODEL
        Simulates both stock price AND volatility changing over time.
        Because volatility depends on the previous day, we must step through time iteratively.
        """
        dt = ttm / self.steps
        
        # Generate two grids of standard normal random numbers
        Z1 = np.random.standard_normal((self.sims, self.steps))
        Z2 = np.random.standard_normal((self.sims, self.steps))
        
        # Create correlated noise (W_S for stock, W_V for volatility)
        W_S = Z1
        W_V = rho * Z1 + math.sqrt(1 - rho**2) * Z2
        
        # Initialize price and variance arrays
        S = np.zeros((self.sims, self.steps + 1))
        V = np.zeros((self.sims, self.steps + 1))
        
        S[:, 0] = market.spot
        V[:, 0] = market.vol ** 2 # Initial variance is vol squared
        
        # We MUST use a loop here because tomorrow's volatility depends on today's volatility
        for t in range(self.steps):
            # We use "Full Truncation" to prevent mathematical errors if variance dips below zero
            v_t_plus = np.maximum(V[:, t], 0)
            
            # Equation 1: Volatility Process (mean-reverting)
            dV = kappa * (theta - v_t_plus) * dt + xi * np.sqrt(v_t_plus * dt) * W_V[:, t]
            V[:, t+1] = v_t_plus + dV
            
            # Equation 2: Stock Process
            dS_log = (market.rate - 0.5 * v_t_plus) * dt + np.sqrt(v_t_plus * dt) * W_S[:, t]
            S[:, t+1] = S[:, t] * np.exp(dS_log)
            
        return S, V
        
    def price_heston_option(self, market: MarketEnvironment, option: OptionContract, kappa, theta, xi, rho):
        """pricing the option using the heston stochastic vol model"""

        paths, _ = self.generate_heston_paths(market, option.ttm, kappa , theta, xi ,rho)
        payoffs = option.calculate_payoff(paths)

        average_payoff =np.mean(payoffs)
        discounted_payoff = math.exp(-market.rate * option.ttm) * average_payoff

        return discounted_payoff, paths

    def generate_correlated_paths(self, spots, rates, vols, corr_matrix, ttm):
        """
        PHASE 3: MULTI-ASSET ENTANGLEMENT (Cholesky Decomposition)
        Simulates multiple correlated assets simultaneously.
        """
        num_assets = len(spots)
        dt = ttm / self.steps
        
        # 1. Cholesky Decomposition: L * L^T = Correlation Matrix
        # L is the "entanglement" matrix. It links the random noise together.
        L = np.linalg.cholesky(corr_matrix)
        
        # We will hold paths for ALL assets. Shape: (Number of Assets, Universes, Days)
        paths = np.zeros((num_assets, self.sims, self.steps + 1))
        for i in range(num_assets):
            paths[i, :, 0] = spots[i]
            
        # 2. Step through time
        for t in range(self.steps):
            # Generate un-correlated random noise for all assets
            Z_uncorrelated = np.random.standard_normal((num_assets, self.sims))
            
            # 3. ENTANGLE THE NOISE using Matrix Multiplication (Dot Product)
            # This mathematically forces the stocks to crash or rally together!
            Z_correlated = L.dot(Z_uncorrelated)
            
            # 4. Apply the correlated physics to each asset
            for i in range(num_assets):
                dS_log = (rates[i] - 0.5 * vols[i]**2) * dt + vols[i] * math.sqrt(dt) * Z_correlated[i, :]
                paths[i, :, t+1] = paths[i, :, t] * np.exp(dS_log)
                
        return paths

class RiskManager:
    """
    Cals the Greak using finite dif methode (bump and revalue)
    tells how much risk they have if the market moves
    """
    def __init__(self, engine: MonteCarloEngine):
        self.engine = engine

    def calculate_greeks(self, market: MarketEnvironment, option: OptionContract):
        #delta(sens to stock price)
        #bump is if the stock price goes up or down by just 1%
        bump_amt = market.spot * 0.01
        
        market_up = MarketEnvironment(market.spot + bump_amt, market.rate, market.vol)
        market_down = MarketEnvironment(market.spot - bump_amt, market.rate, market.vol)

        price_up, _ = self.engine.price_option(market_up, option, use_antithetic=True)
        price_down, _ = self.engine.price_option(market_down, option, use_antithetic=True)

        # delta = rise / run (change in price / change in stock)
        delta = (price_up - price_down) / (2 * bump_amt)

        #vega sens to volatility
        #bump volatility up and down by 1%(0.01)
        vol_bump = 0.01

        market_vol_up = MarketEnvironment(market.spot, market.rate, market.vol + vol_bump)
        market_vol_down = MarketEnvironment(market.spot, market.rate, market.vol - vol_bump)

        price_vol_up, _ = self.engine.price_option(market_vol_up, option, use_antithetic=True)
        price_vol_down, _ = self.engine.price_option(market_vol_down, option, use_antithetic=True)

        # vega = rise / run (change in price / change in volatility)
        vega = (price_vol_up - price_vol_down) / (2 * vol_bump) * 0.01

        return { "Delta":  delta, "Vega": vega}

class Portfolio:
    """Aggregates multiple options and manages risk."""
    def __init__(self, market, engine):
        self.market = market
        self.engine = engine
        self.contracts = []

    def add_contract(self, contract):
        self.contracts.append(contract)

    def calculate_var(self, confidence=0.95):
        """Value at Risk: The potential loss in a worst-case market event."""
        prices = []
        for _ in range(50): # Monte Carlo simulation of portfolio outcome
            val = 0
            for c in self.contracts:
                price, _ = self.engine.price_option(self.market, c, use_antithetic=False)
                val += price
            prices.append(val)
        return np.percentile(prices, (1 - confidence) * 100)
        
    def stress_test(self, spot_shock_pct, vol_shock_pct):
        """
        Simulates a market crash.
        spot_shock_pct: e.g., -0.20 for a 20% market crash.
        vol_shock_pct: e.g., 0.50 for a 50% spike in volatility (panic).
        """
        print(f"\n[!!!] INITIATING STRESS TEST: Spot {spot_shock_pct*100}%, Volatility +{vol_shock_pct*100}% [!!!]")
        
        # 1. Calculate the "Normal" Portfolio Value
        normal_value = 0
        for c in self.contracts:
            price, _ = self.engine.price_option(self.market, c, use_antithetic=True)
            normal_value += price

        # 2. Create the "Doomsday" Market
        crashed_spot = self.market.spot * (1 + spot_shock_pct)
        panicked_vol = self.market.vol * (1 + vol_shock_pct)
        doomsday_market = MarketEnvironment(crashed_spot, self.market.rate, panicked_vol)

        # 3. Calculate the "Doomsday" Portfolio Value
        crashed_value = 0
        for c in self.contracts:
            price, _ = self.engine.price_option(doomsday_market, c, use_antithetic=True)
            crashed_value += price

        # 4. Calculate the Damage
        pnl = crashed_value - normal_value
        print(f"Normal Portfolio Value:   ${normal_value:.2f}")
        print(f"Doomsday Portfolio Value: ${crashed_value:.2f}")
        print(f"TOTAL LOSS (PnL):         ${pnl:.2f}\n")
        return pnl

def run_live_dashboard(market, engine, contract):
    """Animates the Delta risk in real-time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    delta_history = []

    def update(frame):
        # Simulate "Live Market Movement"
        market.spot += np.random.normal(0, 0.5)
        
        # FIX: We use Black-Scholes for instant, smooth live Greeks 
        # instead of Monte Carlo, preventing "Simulation Noise" from breaking the chart!
        d1 = (math.log(market.spot / contract.strike) + (market.rate + 0.5 * market.vol ** 2) * contract.ttm) / (market.vol * math.sqrt(contract.ttm))
        delta = stats.norm.cdf(d1)
        
        delta_history.append(delta)
        if len(delta_history) > 50: delta_history.pop(0)
        
        ax.clear()
        ax.plot(delta_history, color='blue', label="Real-time Delta")
        ax.set_title(f"Live Risk Monitor | Spot: ${market.spot:.2f} | Delta: {delta:.4f}")
        ax.set_ylim(0, 1)

    ani = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.show()

def plot_paths(paths, strike, num_paths=100):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=strike, color="r", linestyle="--", linewidth=2, label=f"Strike Price: (${strike})")
    ax.set_title(f"Monte Carlo Simulation: {num_paths} Universes")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Stock Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Setup limits so the graph doesn't jump around while drawing
    ax.set_xlim(0, paths.shape[1])
    ax.set_ylim(np.min(paths[:num_paths]), np.max(paths[:num_paths]))
    
    # Create empty lines for each universe
    lines = [ax.plot([], [], lw=1.5, alpha=0.5)[0] for _ in range(num_paths)]
    
    plt.show(block=False)
    
    # FIX: Animate the drawing, jumping 5 days per frame for a smooth time-lapse effect
    for i in range(1, paths.shape[1], 5):
        for j, line in enumerate(lines):
            line.set_data(np.arange(i), paths[j, :i])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        
    # Draw the final day to make sure it finishes exactly at the end
    for j, line in enumerate(lines):
        line.set_data(np.arange(paths.shape[1]), paths[j, :])
    fig.canvas.draw()
    fig.canvas.flush_events()

if __name__ == "__main__":
    # Turn on Matplotlib's interactive mode so animations can run smoothly
    plt.ion()
    
    print("-- ENTERPRISE PRICING ENGINE --\n")

    # Fetch real data for a stock
    ticker = "AAPL"
    try:
        spot, vol = get_market_data(ticker)
        print(f"Data retrieved for {ticker}: Spot={spot:.2f}, Vol={vol:.2f}")
    except:
        print("Could not fetch data, using default values.")
        spot, vol = 100.0, 0.20

    current_market = MarketEnvironment(spot_price=spot, risk_free_rate=0.05, volatility=vol)
    
    euro_option = EuropeanCall(strike_price=spot * 1.05, time_to_maturity=1.0)
    asian_option = AsianCall(strike_price=spot * 1.05, time_to_maturity=1.0)
    
    engine = MonteCarloEngine(simulations=100_000, steps=252)
    my_portfolio = Portfolio(current_market, engine)
    my_portfolio.add_contract(euro_option)

    # Ground Truth
    bs_price = euro_option.black_scholes_anly(current_market)
    print(f"Exact theo price (BS): ${bs_price:.4f}\n")

    # Standard Engine Test
    print("Running Standard Engine...")
    start_time = time.time()
    std_price, _ = engine.price_option(current_market, euro_option, use_antithetic=False)
    std_time = time.time() - start_time
    std_error = abs(std_price - bs_price)
    print(f"Standard Price:   ${std_price:.4f} | Error: {std_error:.4f} | Time: {std_time:.4f}s\n")

    # Antithetic Engine
    print("Running Antithetic (Mirror) Engine...")
    start_time = time.time()
    anti_price, anti_paths = engine.price_option(current_market, euro_option, use_antithetic=True)
    anti_time = time.time() - start_time
    anti_error = abs(anti_price - bs_price)
    print(f"Antithetic Price: ${anti_price:.4f} | Error: {anti_error:.4f} | Time: {anti_time:.4f}s\n")

    # Exotic Option Test
    print("Pricing Exotic Asian Option...")
    asian_price, _ = engine.price_option(current_market, asian_option, use_antithetic=True)
    print(f"Asian Option Price: ${asian_price:.4f} (Cheaper due to averaging!)\n")

    # Calculate Variance Reduction
    error_red = ((std_error - anti_error) / std_error) * 100 if std_error > 0 else 0
    print(f"Variance Reduction: Error reduced by {error_red:.1f}%!\n")

    # Risk management (the greeks)
    print("risk management")
    risk_desk = RiskManager(engine)
    greeks = risk_desk.calculate_greeks(current_market, euro_option)
    print(f"Delta: {greeks['Delta']:.4f} ( Option gains $ {greeks['Delta']:.2f} for every 1% change in stock price/it goes up )")
    print(f"Vega: {greeks['Vega']:.4f} (Option gains $ {greeks['Vega']:.2f} for every 1% change in volatility)")

    # Portfolio VaR
    print(f"\nPortfolio VaR (95%): ${my_portfolio.calculate_var():.2f}")
    
    # Run the Stress Test (e.g., 1987 Black Monday scenario)
    my_portfolio.stress_test(spot_shock_pct=-0.20, vol_shock_pct=0.50)

    print("\n--- PHASE 2: HESTON MODEL PRICING ---")
    """here:
    kappa: mean reversion speed( how fast volatility returns to normal)
    theta: long term avg of variance
    xi: (vol  of vol): how volatile is the volatilty itslef?
    rho: correlation(-0.7 means when the market drops, vol spikes)
    """
    heston_price, heston_paths = engine.price_heston_option(
        current_market, euro_option,
        kappa = 2.0, theta = vol**2, xi= 0.2, rho = -0.7
    )
    print(f"Heston model price(Stochastic Vol): ${heston_price:.4f}")
    print(f"Exact theo price (BS): ${bs_price:.4f}")

    print("\n--- PHASE 3: MULTI-ASSET CORRELATION (CHOLESKY) ---")
    # Let's pretend we have 2 tech stocks (AAPL and MSFT) that move together 80% of the time.
    basket_spots = [150.0, 200.0]
    basket_vols = [0.20, 0.25]
    basket_rates = [0.05, 0.05]
    
    # Correlation Matrix (1.0 on the diagonal because a stock is 100% correlated with itself)
    correlation_matrix = np.array([
        [1.0, 0.8],
        [0.8, 1.0]
    ])
    
    multi_paths = engine.generate_correlated_paths(basket_spots, basket_rates, basket_vols, correlation_matrix, ttm=1.0)
    print(f"Successfully generated {engine.sims} correlated universes for {len(basket_spots)} assets!")
    
    # Let's prove the math works by calculating the actual correlation of our simulated final prices!
    final_prices_asset_A = multi_paths[0, :, -1]
    final_prices_asset_B = multi_paths[1, :, -1]
    empirical_corr = np.corrcoef(final_prices_asset_A, final_prices_asset_B)[0, 1]
    print(f"Target Input Correlation:      0.8000")
    print(f"Actual Simulated Correlation: {empirical_corr:.4f} (The Cholesky Math Works!)\n")

    print("Rendering charts...")
    plot_paths(heston_paths, euro_option.strike, num_paths=100)

    print("\nStarting Live Dashboard...")
    # Turn OFF interactive mode right before the final loop so the animation window stays open
    plt.ioff() 
    run_live_dashboard(current_market, engine, euro_option)