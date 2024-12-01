import scipy.stats 
import numpy as np
import yfinance as yf # type: ignore
import pandas as pd # type: ignore

def sr(portfolio):
    """
    Computes the sharpe ratio, a measure of the quality of a portfolio's returns.
    
    Parameters:
    portfolio (pandas Series): the daily returns of the portfolio

    Returns:
    float: the sharpe ratio
    """
    rf_rate = (yf.download('^TNX', interval='1d', start='2018-01-01')['Adj Close'])/100
    rf_rate = rf_rate.dropna()/rf_rate.size
    return (portfolio.mean() - rf_rate.mean())*rf_rate.size / (portfolio.std()*np.sqrt(rf_rate.size))
def sd_sr(portfolio):
    
    """
    Computes the standard deviation of the probabilistic sharpe ratio, a measure of the quality of
    a portfolio's returns. It takes into account the skewness and kurtosis of
    the returns, as well as the number of observations. For a portfolio with
    normally distributed returns, this will equal the standard sharpe ratio.
    
    Parameters:
    portfolio (pandas Series): the daily returns of the portfolio

    Returns:
    float: the st. deviation of the probabilistic sharpe ratio
    """
    skew = scipy.stats.skew(portfolio,axis=0,bias=False)
    kurt = scipy.stats.kurtosis(portfolio,axis=0,bias=False)
    n_obs = portfolio.size
    reg_sr = sr(portfolio)

    return np.sqrt(1-skew*reg_sr+(kurt-1)/4*reg_sr**2/(n_obs-1))
def prob_sharpe(bench_sr, portfolio):
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
    sr_diff = sr(portfolio) - bench_sr
    sr_vol = sd_sr(portfolio)
    psr = scipy.stats.norm.cdf(sr_diff/sr_vol)

    return psr