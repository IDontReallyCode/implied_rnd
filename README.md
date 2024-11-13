# implied_rnd
Package to extract the implied Risk-Neutral Density from a set of option quotes

## General use
Call the impliedrnd() function with:
- array of [strikes, mid-quotes for OTM/ATM calls and puts]
- time in years
- interpolation method (choice from a list)
- extrapolation method (choice from a list)
- range and number of points for output

Receive:
- array with values of S and densities




## PROJECT 002 - Simply visuallize the fit, and possibly the extrapolation
Poly M4 fitting using OLS or WOLS goes well for intrapolation, but extrapolation is horrible, obviously

SVI fitting using optimization is questionnable. 
### SVI fitting
1. Good starting values are necessary
2. Multiple starting values might be necessary, but I have not tested this yet. (Need to look up old code with ranges for parameters)
3. Hard boundaries on parameters are necessary
4. Penalties on non-linear constraints are necessary as well
5. Without a good weighting scheme, it seems the fitting my result in pretty bad fits. However, it is not clear whether this is caused from fitting the wrong region of the data or from bad strating values.

#### Potential problem
- Without data on the left side (positive returns) of the IVar curve, it might be impossible to identify the skew
- Perhaps, we need to keep the unreliable/noisy data in there to get some control on the left side?


#### IDEAS TO FIX
1. Dig up the NGARCH fitting code with multiple starting values
2. Re-parametrize such that the positive condition is a parameter. This way, it will be included in the hard boundaries on parameters instead.
3. Come up with a strong weighting scheme
4. Re-parametrize in terms on right slope.
    1. Estimate that slope from the data and provide a confidence interval as the limit on that parameter
    2. Estimate the region where the minimum might lie, provided limit on that parameter
    3. 



## PROJECT 004 - Compare IVS fit on RMSE

### Description
The idea here is to compare the IVS fit of three basic models:
1. A simple polynomial of order 2 including a cross-product fitted on IVolS
2. A simple polynomial of order 2 including a cross-product fitted on IVarS
3. Francois, Galarneau-Vincent, Gauthier, & Godin 2022 on IVolS
4. SVI augmented with term-structure factors and cross-products fitted on IVarS


### Methodology
1. [TODO] Randomly pick equity tickers with highly liquid options (insert link to SQL query)
2. Randomly pick EOD timestamps
3. Fit all four models above and get the RMSE over the surface
4. Gather the statistics for the RMSE for all four models
    a. mean RMSE
    b. stdev RMSE
    c. min RMSE
    d. max RMSE

Then repeat for SPX and SPY


## PROJECT 005 - Compare RND, 
### Description


### Methodology