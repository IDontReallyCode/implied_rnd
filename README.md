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
    1. $a0 + a2 * (a1 * (x - b0) + np.sqrt(b1^2 + (x - b0)^2)) - a2*b1*np.sqrt(1-a1^2)$
3. Come up with a strong weighting scheme
4. Re-parametrize in terms on right slope.
    1. Estimate that slope from the data and provide a confidence interval as the limit on that parameter
    2. Estimate the region where the minimum might lie, provided limit on that parameter
    3. 


## 2024-11-13 UPDATE
1. I changed the specification of the model to $a_0 + a_2 * (a_1 * (x - b_0) + np.sqrt(b_1^2 + (x - b_0)^2)) - a_2*b_1*np.sqrt(1-a_1^2)$
2. I changed the optimization to use the bounds $a_0>0$, $-1 < a_1 < 1$, $a_2 > 0$, and $b_1 > 0$

The optimization is now easier. We still have issues.
1. If we use the filtered data as in teh literature, the fit is good, but the extrapolation towards negative log-moneyness might be quite off.
2. If we use "less" filtered data, the fit is not better.
3. If we use bid-ask iv, the fit is not better.
4. There is sometimes a compromise between fitting the tails better, or fitting the center of the distribution better. Sometimes, it does not matter, but sometimes it does.
    1. I considered using different weighting scheme. The problem is that we may end-up with worse fit.

For now, I will simply continue with what I have. The code for extrapolation on the log-moneyness already works and the RND extraction works well too.

### Problem remaining for NOW
The tail extension using the Generalized Pareto is still fucked up and need to be fixed

#### Problem remaining for later
We can still improve the over the SVI. Perhaps we can have two SVI.
$a_0 + a_2 * (a_1 * (x - b_0) + np.sqrt(b_1^2 + (x - b_0)^2)) - a_2*b_1*np.sqrt(1-a_1^2)  + a_4 * (a_3 * (x - b_2) + np.sqrt(b_3^2 + (x - b_2)^2)) - a_4*b_3*np.sqrt(1-a_3^2)$
using a similar set of bounds on parameters. 

We could, instead, add some functions with asymptotics to zero. Like a normal density, for example. Completing the model with more factors will need to be done carefully to preserve the non-negative variance, but there are plenty of ideas to try.


## PROJECT 003 - compare the RND extraction

### PROJECT 003 - 001: No extrapolation
So, here, we fit on the data that is available, we interpolate, and then apply Breenden Litzenberger to get the RND, as is. We do not extend it, and we do not scale it to 1. We simply assume that the data is good to get the density where it is at right now.

The sample data test shows there can be significant differences between the weighted polynomial of order 4 and the SVI model with the new specification.


### PROJECT 003 - 002: Extrapolation
Here, we compare extrapolation from the log-moneyness usign the SVI specification and the polynomial of order 4 with the Generalized Pareto tail extension.

### TODO
- [x] Fix the left tail estimation which is all screwed up.
    - [x] Make sure we have the right data on the left tail
    - [x] Flip the x-axis data, and make sure we have it in the right order
    - [x] Fit the model with either 2 or 3 parameters on either 2 or 3 points
    - [x] Make sure the extrapolation is correct.
- [x] verify the fit and extrapolation on the right side.
- [ ] Now, create a new one estimation function to fit both sides at once, while keeping the integral sum up to 1

#### Double estimation 
f(x) = fitdistribution(xfit, yfit, xfull)

Need a function that uses a theta of 6 elements
First three to fit the left tail
Last three to fit the right tail
We get the quality of the fit
Also, extrapolate and intergrat to see what it integrates too
Add up the error on all 4 points and the difference between the integral and 1




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