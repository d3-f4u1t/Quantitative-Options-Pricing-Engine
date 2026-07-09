# Quantitative Options Pricing Engine

This project provides a high-performance Monte Carlo simulation engine for pricing European financial options. It implements a vectorized approach in Python using NumPy to achieve significant computational speedup over traditional iterative methods.

The primary goal is to offer a tool that is both performant enough for quantitative analysis and clear enough to serve as an educational resource for understanding the principles of stochastic calculus in finance and the benefits of vectorization.

### What this project does
*   **Models Asset Price Dynamics**: Simulates future stock price paths using the Geometric Brownian Motion (GBM) stochastic differential equation.
*   **Prices European Options**: Calculates the fair value of a European call option by simulating thousands of potential future outcomes.
*   **Implements High-Performance Simulation**: Leverages NumPy vectorization to execute the entire Monte Carlo simulation without explicit `for` loops, enabling the rapid calculation of tens of millions of stochastic data points.
*   **Provides Analytical Benchmarking**: Includes an implementation of the closed-form Black-Scholes model to serve as a benchmark for the accuracy of the Monte Carlo simulation.
*   **Visualizes Stochastic Paths**: Generates plots of the simulated asset price paths to provide a visual representation of the market's probabilistic nature.

### Project Structure
The project is currently contained within a single script for simplicity. A recommended scalable structure for future development is also presented.

**Current Structure:**
```
OptionPricingEngine/
|-- option_Eng.py
`-- requirements.txt
```


### How the Simulator Works
The engine prices an option by simulating a large number of possible future price paths for the underlying asset and then calculating the average payoff of the option in those futures.

For each simulation run:

1.  **Path Generation**: The future asset price is modeled using the discretized solution to the Geometric Brownian Motion SDE:
    $$S_{t+\Delta t} = S_t \times \exp\left( \left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right)$$
    Instead of looping through time steps and simulations, the engine generates a `(simulations, steps)` matrix of standard normal random variates ($Z$) and applies the formula across the entire grid simultaneously.

2.  **Payoff Calculation**: At the expiration of the option, the payoff for each simulated path is calculated. For a European call option, this is:
    $$ \text{Payoff} = \max(S_T - K, 0) $$
    where $S_T$ is the final asset price and $K$ is the strike price.

3.  **Discounting**: The average of all simulated payoffs is computed and then discounted back to its present value using the risk-free rate `r`:
    $$ \text{Option Price} = \mathbb{E}[\text{Payoff}] \times e^{-rT} $$

This vectorized process allows for the simulation of hundreds of thousands of paths over hundreds of time steps in under a second.

### Quick Start
To run the engine and reproduce the analysis, follow these steps.

1.  **Install dependencies:**
    ```bash
    python -m pip install -r requirements.txt
    ```

2.  **Run the simulation:**
    ```bash
    python option_Eng.py
    ```

### Output
The script will first print the analytical price derived from the Black-Scholes model. It then runs the Monte Carlo simulation, printing the resulting option price and the total computation time. Finally, it will render a plot visualizing a subset of the simulated price paths.

**Example Console Output:**
```
--PRICING ENGIN--
EXact theo price (BS): 7.9653

Runing Qunat Engin for 100100
Vectorized Price: 7.9521
Vectorized Time: 0.8912 seconds

Rendering charts
```

### Research Direction
This initial version establishes a robust, vectorized foundation for pricing European options. Future work can extend the engine's capabilities to align it with more advanced research and practical applications.

Phase 2 development could include:
*   **Variance Reduction Techniques**: Implementing methods like Antithetic Variates and Control Variates to increase simulation accuracy without increasing computational load.
*   **Exotic Options Pricing**: Extending the model to price path-dependent options, such as Asian or Barrier options, by leveraging the full set of path data.
*   **Alternative Stochastic Processes**: Incorporating more complex models like the Heston model (stochastic volatility) or Merton's jump-diffusion model.
*   **Parameterization and Configuration**: Moving market parameters to an external configuration file (e.g., JSON) to allow for easier scenario testing and analysis.
