import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def monteCarlo(portfolio, sims, time, intial_value):
    """
    Perform a Monte Carlo simulation of a portfolio of assets.
    
    Parameters
    ----------
    portfolio : pandas.DataFrame
        A pandas DataFrame of historical returns of the assets in the portfolio
    sims : int
        The number of simulations to run
    time : int
        The number of days to simulate
    intial_value : float
        The initial value of the portfolio
        
    Returns
    -------
    A plot of the simulated portfolio values
    """
    
    returns = portfolio.pct_change()
    returns = portfolio.dropna()
    cov = returns.cov()
    returns_mean = returns.mean()
    weights = [1/len(portfolio.columns)]*len(portfolio.columns)

    meanM = np.full(shape=(time, len(weights)), fill_value=returns_mean)
    meanM = meanM.T
    portfolio_sims = np.full(shape=(time, sims), fill_value=0.0)

    for m in range(0, sims):
        Z = np.random.normal(size=(time, len(weights)))
        L = np.linalg.cholesky(cov)
        daily_ret = meanM + np.inner(L, Z)
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, daily_ret.T) + 1)*intial_value
    plt.figure(figsize=(10,6))
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value (USD$)')
    plt.xlabel('Days')
    plt.title('MC simulation of the portfolio')
    plt.show()

