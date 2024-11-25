import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import genpareto
import pandas as pd
import optimizing as opt
from scipy.integrate import simps


# Load your data
outputx = np.load('v1_outputx.npy')
outputf = np.load('v1_outputf.npy')
extlftmask = np.load('v1_extlftmask.npy')
extrgtmask = np.load('v1_extrgtmask.npy')
interpmask = np.load('v1_interpmask.npy')

# The data that does not change, determined by Breenden and Litzenberger (1978)
xinterp = outputx[interpmask]
yinterp = outputf[interpmask]
# Left tail **************************************************
# the x data to be plotted
xlefttail = outputx[extlftmask]
# the x data to be fitted
xlefttailfit = outputx[interpmask][0:2][::-1]
refpoint = outputx[interpmask][1]
xlefttailfit = -1*(xlefttailfit - refpoint)
ylefttailfit = outputf[interpmask][0:2][::-1]

xlefttaileval = -1*(xlefttail - refpoint)


# Let us fit the scale and shape parameters of the Generalized Pareto Distribution, leaving location = 0
thetaleft = opt._fittail(xlefttailfit, ylefttailfit)

# Right tail **************************************************

# xlefttailin = -1*(xlefttail - refpoint)
# xrighttail = outputx[extrgtmask]
# # refpoint = outputx[interpmask][-1]
# xrighttailin = xrighttail - refpoint
# the x data to be plotted
xrighttail = outputx[extrgtmask]

xrighttailfit = outputx[interpmask][-2:]
refpoint = outputx[interpmask][-2]
xrighttailfit = xrighttailfit - refpoint
yrighttailfit = outputf[interpmask][-2:]
xrighttaileval = outputx[extrgtmask] - refpoint

thetarigt = opt._fittail(xrighttailfit, yrighttailfit)

# Calculate the area under the curve
midlpartarea = simps(yinterp, xinterp)



# Define the functions using Generalized Pareto Distribution
def func1(x, c, loc, scale):
    return genpareto.pdf(x, c, loc, scale)

def func2(x, c, loc, scale):
    return genpareto.pdf(x, c, loc, scale)

# Initial parameters
cleft_init, locleft_init, scaleleft_init = thetaleft[0], 0, thetaleft[1]
criht_init, locriht_init, scaleriht_init = thetarigt[0], 0, thetarigt[1]

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)
line_data, = plt.plot(xinterp, yinterp, label='Data')
ax.set_ylim(0, max(yinterp) + 0.0001)
line1, = plt.plot(xlefttail, func1(xlefttaileval, cleft_init, locleft_init, scaleleft_init), label='Left tail')
line2, = plt.plot(xrighttail, func2(xrighttaileval, criht_init, locriht_init, scaleriht_init), label='Right tail')
plt.legend()
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Interactive Plot with Sliders')

# Define sliders
axcolor = 'lightgoldenrodyellow'
ax_c1 = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_loc1 = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_scale1 = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_c2 = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_loc2 = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_scale2 = plt.axes([0.1, 0, 0.65, 0.03], facecolor=axcolor)

s_c1 = Slider(ax_c1, 'c1', thetaleft[0]/5, thetaleft[0]*5, valinit=cleft_init)
s_loc1 = Slider(ax_loc1, 'loc1', -10.0, 0, valinit=locleft_init)
s_scale1 = Slider(ax_scale1, 'scale1', thetaleft[1]/5, thetaleft[1]*5, valinit=scaleleft_init)
s_c2 = Slider(ax_c2, 'c2', thetarigt[0]/10, thetarigt[0]*100, valinit=criht_init)
s_loc2 = Slider(ax_loc2, 'loc2', -20.0, 0, valinit=locriht_init)
s_scale2 = Slider(ax_scale2, 'scale2', thetarigt[1]/100, thetarigt[1]*10, valinit=scaleriht_init)

# Update function
def update(val):
    c1 = s_c1.val
    loc1 = s_loc1.val
    scale1 = s_scale1.val
    c2 = s_c2.val
    loc2 = s_loc2.val
    scale2 = s_scale2.val
    # leftint = opt.genpareto.cdf(min(xlefttaileval), c=c1, loc=loc1, scale=scale1)
    # rigtint = opt.genpareto.cdf(min(xrighttaileval), c=c2, loc=loc2, scale=scale2)
    # get both tails int by simps instead
    leftint = simps(func1(xlefttaileval, c1, loc1, scale1), xlefttail)
    rigtint = simps(func2(xrighttaileval, c2, loc2, scale2), xrighttail)
    line1.set_ydata(func1(xlefttaileval, c1, loc1, scale1))
    line2.set_ydata(func2(xrighttaileval, c2, loc2, scale2))
    total = leftint + midlpartarea + rigtint
    ax.set_title(f'Left Tail CDF: {leftint:.4f}, Right Tail CDF: {rigtint:.4f}, Total CDF: {total:.4f}')
    fig.canvas.draw_idle()

# Call update function on slider value change
s_c1.on_changed(update)
s_loc1.on_changed(update)
s_scale1.on_changed(update)
s_c2.on_changed(update)
s_loc2.on_changed(update)
s_scale2.on_changed(update)

plt.show()