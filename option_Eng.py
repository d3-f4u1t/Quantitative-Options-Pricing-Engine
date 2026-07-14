import numpy as np
import time
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

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

class RiskManager:
    """
    Cals the Greak usingh finite dif methode (bump and revalue)
    tells how much risk they have if the market moves
    """

    def __init__(self, engin: MonteCarloEngine):
        self.engine = engine

    def calculate_greeks(self, market: MarketEnvironment, option: OptionContract):
        #delta(sens to stock price)
        #bump is if the stock price goes upor down by just 1%

        bump_amt = market.spot * 0.01
        
        market_up = MarketEnvironment(market.spot + bump_amt, market.rate, market.vol)
        market_down = MarketEnvironment(market.spot - bump_amt, market.rate, market.vol)

        price_up, _ = self.engine.price_option(market_up, option, use_antithetic=True)
        price_down, _ = self.engine.price_option(market_down, option, use_antithetic=True)

        # delta = rise / run (change in price/ change in stock)
        delta = (price_up - price_down) / (2 * bump_amt)

        #vega sens to volatility
        #bump volatility up and down by 1%(0.01)
        vol_bump = 0.01

        market_vol_up = MarketEnvironment(market.spot, market.rate, market.vol + vol_bump)
        market_vol_down = MarketEnvironment(market.spot, market.rate, market.vol - vol_bump)

        price_vol_up, _ = self.engine.price_option(market_vol_up, option, use_antithetic=True)
        price_vol_down, _ = self.engine.price_option(market_vol_down, option, use_antithetic=True)

        # vega = rise / run (change in price/ change in volatility)
        vega = (price_vol_up - price_vol_down) / (2 * vol_bump) * 0.01

        return { "Delta":  delta, "Vega": vega}



        #


def plot_paths(paths, strike, num_paths=100):
    plt.figure(figsize=(12, 6))
    # plots the rows
    plt.plot(paths[:num_paths].T, lw=1.5, alpha=0.5)
    #red dashed line for target strick price
    plt.axhline(y=strike, color="r", linestyle="--", linewidth=2, label=f"Strike Price: (${strike})")
    
    plt.title(f"Monte Carlo Simulation: {num_paths} Universes")
    plt.xlabel("Trading days")
    plt.ylabel("Stock Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    print("-- ENTERPRISE PRICING ENGINE --\n")

    # 1. Create objects
    current_market = MarketEnvironment(spot_price=100.00, risk_free_rate=0.05, volatility=0.20)
    
    euro_option = EuropeanCall(strike_price=105.00, time_to_maturity=1.0)
    asian_option = AsianCall(strike_price=105.00, time_to_maturity=1.0)
    
    engine = MonteCarloEngine(simulations=100_000, steps=252)

    #Ground Truth
    bs_price = euro_option.black_scholes_anly(current_market)
    print(f"Exact theo price (BS): ${bs_price:.4f}\n")

    #Standard Engine Test
    print("Running Standard Engine...")
    start_time = time.time()
    std_price, _ = engine.price_option(current_market, euro_option, use_antithetic=False)
    std_time = time.time() - start_time
    std_error = abs(std_price - bs_price)
    print(f"Standard Price:   ${std_price:.4f} | Error: {std_error:.4f} | Time: {std_time:.4f}s\n")

    # 4 Antithetic Engine
    print("Running Antithetic (Mirror) Engine...")
    start_time = time.time()
    anti_price, anti_paths = engine.price_option(current_market, euro_option, use_antithetic=True)
    anti_time = time.time() - start_time
    anti_error = abs(anti_price - bs_price)
    print(f"Antithetic Price: ${anti_price:.4f} | Error: {anti_error:.4f} | Time: {anti_time:.4f}s\n")

    # 5. Exotic Option Test
    print("Pricing Exotic Asian Option...")
    asian_price, _ = engine.price_option(current_market, asian_option, use_antithetic=True)
    print(f"Asian Option Price: ${asian_price:.4f} (Cheaper due to averaging!)\n")

    # Calculate Variance Reduction
    error_red = ((std_error - anti_error) / std_error) * 100 if std_error > 0 else 0
    print(f"Variance Reduction: Error reduced by {error_red:.1f}%!\n")

    # Risk managment ( the greeks)

    print("risk managment")

    risk_desk = RiskManager(engine)
    greeks = risk_desk.calculate_greeks(current_market, euro_option)

    print(f"Delta: {greeks['Delta']:.4f} ( Option gains $ {greeks['Delta']:.2f} for every 1% change in stock price/it goes up )")
    print(f"Vega: {greeks['Vega']:.4f} (Option gains $ {greeks['Vega']:.2f} for every 1% change in volatility)")



    print("Rendering charts...")
    plot_paths(anti_paths, euro_option.strike, num_paths=100)