import numpy as np
import statsmodels.api as sm

# Generate some example data
np.random.seed(0)
x0 = np.random.rand(100)
x1 = np.random.rand(100)
y = 2 * x0 + 3 * x1 + np.random.randn(100)

# Assign different weights to x0 and x1 for all observations
weights = np.column_stack((np.full(100, 2.0), np.full(100, 1.0)))

# Create a design matrix
X = np.column_stack((x0, x1))

# Fit a weighted OLS regression model
model = sm.WLS(y, X, weights=1/weights)  # Note: inverse of weights
result = model.fit()

# Print the regression results
print(result.summary())
