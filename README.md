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
