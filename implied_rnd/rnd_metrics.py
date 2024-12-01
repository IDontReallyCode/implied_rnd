import numpy as np
from scipy.stats import norm
import statsmodels.api as sm


def expected_value(x, f):
    return np.sum(x * f)


def variance(x, f):
    mean = expected_value(x, f)
    return np.sum((x - mean) ** 2 * f)


def skewness(x, f):
    mean = expected_value(x, f)
    var = variance(x, f)
    return np.sum((x - mean) ** 3 * f) / var ** 1.5


def kurtosis(x, f):
    mean = expected_value(x, f)
    var = variance(x, f)
    return np.sum((x - mean) ** 4 * f) / var ** 2


def value_at_risk(x, f, alpha):
    cumulative = np.cumsum(f)
    return x[np.searchsorted(cumulative, alpha)]


def expected_shortfall(x, f, alpha):
    var = value_at_risk(x, f, alpha)
    return np.sum(x[x <= var] * f[x <= var]) / np.sum(f[x <= var])


# # Example usage
# if __name__ == "__main__":
#     x = np.array([1, 2, 3, 4, 5])
#     f = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    
#     print("Expected Value:", expected_value(x, f))
#     print("Variance:", variance(x, f))
#     print("Skewness:", skewness(x, f))
#     print("Kurtosis:", kurtosis(x, f))
#     print("Value-at-Risk (95%):", value_at_risk(x, f, 0.95))
#     print("Expected Shortfall (95%):", expected_shortfall(x, f, 0.95))