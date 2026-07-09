import numpy as np
import time
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

# market parameters 

INITIAL_STOCK_PRICE = 100.00 #CURRENT PRICE
STRIKE_PROCE = 105.00 #TRAGET PRICE. MONEY IS ONLY MADE IF THE STOCK GOES SBOVE THIS PRICE
TIME_TO_MATURITY = 1.0 #WE ARE SIMULATING 1 YEAR INTO THE FUTURE
RISK_FREE_RATE = 0.05 #BASELINE INTEREST RATE (5%)
VOLATILITY = 0.20 #HOW MUCH UP AND DOWN DOES A STOCKDO

SIMULATIONS = 100_100 #SIMULATION OF 100,000
STEPS = 252 #TRADING DAYS IN A SIMULATED YEAR


def black_scholes_anly():

    #d1 and d2 are complex probability boundaries in the formula

    d1 = (math.log(INITIAL_STOCK_PRICE / STRIKE_PROCE) + (RISK_FREE_RATE + 0.5 * VOLATILITY ** 2) * TIME_TO_MATURITY) / (VOLATILITY * math.sqrt(TIME_TO_MATURITY))
    d2 = d1 - VOLATILITY * math.sqrt(TIME_TO_MATURITY)

    #cal the final theo price
    call_price = INITIAL_STOCK_PRICE * stats.norm.cdf(d1) - STRIKE_PROCE * math.exp(-RISK_FREE_RATE * TIME_TO_MATURITY) * stats.norm.cdf(d2)

    return call_price


def Monte_carlo_vectorized():
    #dt is our time step
    dt = TIME_TO_MATURITY / STEPS
    
    # Grid this will gen 25 mil random stocks
    #and a grid of 100,000 rows and 252 coloumn
    

    z = np.random.standard_normal((SIMULATIONS, STEPS))

    #cal daily growth for the entiner grid

    growth_factors = np.exp((RISK_FREE_RATE - 0.5 * VOLATILITY ** 2) * dt + VOLATILITY * math.sqrt(dt) * z)
    

    #timeline for linking the days togther
    #axices = 1 multipies left to right across column
    price_paths = INITIAL_STOCK_PRICE * np.cumprod(growth_factors, axis = 1)

    #final price

    final_prices = price_paths[:, -1]

    #cal payoff- did we beat the strick price? if not then the payoff is 0

    payoffs = np.maximum(final_prices - STRIKE_PROCE, 0)

    #avg

    average_payoff = np.mean(payoffs)
    discounted_payoff = math.exp(-RISK_FREE_RATE * TIME_TO_MATURITY) * average_payoff


    return discounted_payoff, price_paths


def plot_paths(paths, num_paths = 100):

    plt.figure(figsize = (12, 6))
    
    #plots the first 100 rows

    plt.plot(paths[:num_paths].T, lw = 1.5, alpha = 0.5)

    #draw a red dashed line for our target strick price

    plt.axhline(y = STRIKE_PROCE, color = "r", linestyle  = "--", linewidth = 2, label = f"Strike Price: (${STRIKE_PROCE})")

    #LABELS:
    plt.title(f"Monte carlo Simulation: {num_paths}")
    plt.xlabel("Trading days")
    plt.ylabel("Stock Price ($)")
    plt.legend()
    plt.grid(True, alpha = 0.3)

    #Show chats

    plt.show()



    






if __name__  == "__main__":
    print("--PRICING ENGIN--")

    bs_price = black_scholes_anly()
    print(f"EXact theo price (BS): {bs_price:.4f}\n")

    #vector simulation

    print(f"Runing Qunat Engin for {SIMULATIONS}")
    start_time = time.time()
    mc_price, price_paths = Monte_carlo_vectorized()
    fast_time = time.time() - start_time

    print(f"Vectorized Price: {mc_price:.4f}")
    print(f"Vectorized Time: {fast_time:.4f} seconds\n")

    print("Rendering charts")
    plot_paths(price_paths, num_paths = 150)
