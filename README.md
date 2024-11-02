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