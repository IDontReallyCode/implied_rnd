import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import genpareto

# Load the numpy arrays
tailx = np.load('tailx.npy')
tailf = np.load('tailf.npy')
# tailx = tailx - tailx[2]

# Scatter plot of tailx and tailf
fig, ax = plt.subplots()
plt.scatter(tailx, tailf, label='Data Points')
plt.xlabel('tailx')
plt.ylabel('tailf')
# plt.ylim(min(tailf) - 0.1 * abs(min(tailf)), max(tailf) + 0.1 * abs(max(tailf)))
plt.title('Scatter Plot of tailx and tailf')

# Initial parameters for genpareto
c0, loc0, scale0 = 0.5, 0, 1

# Plot the initial Genpareto.pdf
x_values = np.linspace(min(tailx), max(tailx), 100)
pdf_initial, = plt.plot(x_values, genpareto.pdf(x_values, c0, loc0, scale0), 'r-', lw=2, label='GenPareto PDF')
plt.legend()

# Adjust the main plot to make room for sliders
plt.subplots_adjust(left=0.1, bottom=0.35)

# Add sliders for the parameters
axcolor = 'lightgoldenrodyellow'
ax_c = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor=axcolor)
ax_loc = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=axcolor)
ax_scale = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)

s_c = Slider(ax_c, 'c', 0.01, 55.0, valinit=c0)
s_loc = Slider(ax_loc, 'loc', min(tailx)-1, max(tailx)+1, valinit=loc0)
s_scale = Slider(ax_scale, 'scale', 0.1, 50.0, valinit=scale0)

# Update function to redraw the plot with new parameters
def update(val):
    c = s_c.val
    loc = s_loc.val
    scale = s_scale.val
    pdf_initial.set_ydata(genpareto.pdf(x_values, c, loc, scale))
    fig.canvas.draw_idle()

# Attach the update function to the sliders
s_c.on_changed(update)
# s_loc.on_changed(update)
s_scale.on_changed(update)

plt.show()
