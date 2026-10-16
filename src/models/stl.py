"""STL decomposition extracted from STL-code.txt."""
from statsmodels.tsa.seasonal import STL


def decompose(series, period=12, robust=True):
    return STL(series, period=period, robust=robust).fit()
