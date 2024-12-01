import scipy.stats 
import numpy as np
import yfinance as yf # type: ignore
import pandas as pd # type: ignore
def sd_sr(portfolio):
    
    """
    Computes the daily standard deviation of the probabilistic sharpe ratio, a measure of the quality of
    a portfolio's returns. It takes into account the skewness and kurtosis of
    the returns, as well as the number of observations. For a portfolio with
    normally distributed returns, this will equal the standard sharpe ratio.
    
    Parameters:
    portfolio (pandas Series): the daily returns of the portfolio

    Returns:
    float: the daily st. deviation of the probabilistic sharpe ratio
    """
    skew = scipy.stats.skew(portfolio,axis=0,bias=False)
    kurt = scipy.stats.kurtosis(portfolio,axis=0,bias=False)
    n_obs = portfolio.size
    rf_rate = (yf.download('^TNX', interval='1d', start='2018-01-01')['Adj Close'])/100
    rf_rate = rf_rate.dropna()/252
    #portfolio = portfolio.to_numpy()
    reg_sr = (portfolio.mean() - rf_rate.mean()) / portfolio.std()
    print(type(rf_rate))
    return np.sqrt(1-skew*reg_sr+(kurt-1)/4*reg_sr**2/(n_obs-1))
def prob_sharpe(bench_sr, sr, portfolio):
    """
    Computes the probabilistic sharpe ratio, a measure of the quality of a portfolio's returns which takes into
    account the skewness and kurtosis of the returns, as well as the number of observations. The probabilistic sharpe
    ratio is the z-score of the difference between the portfolio's sharpe ratio and a benchmark sharpe ratio, 
    standardized by the standard deviation of the probabilistic sharpe ratio.
    
    Parameters:
    bench_sr (float): the sharpe ratio of the benchmark
    sr (float): the sharpe ratio of the portfolio
    portfolio (pandas Series): the daily returns of the portfolio
    
    Returns:
    float: the probabilistic sharpe ratio
    """
    sr_diff = sr - bench_sr
    sr_vol = sd_sr(portfolio)
    psr = scipy.stats.norm(sr_diff/sr_vol)
    return psr