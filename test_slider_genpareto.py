import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import genpareto

# Setting up the plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.35)

# Initial parameters for genpareto
c0, loc0, scale0 = 1.0, 0.0, 1.0

# Define the range for x values
x_values = np.linspace(-2, 8, 1000)

# Plot the initial Genpareto.pdf
pdf_initial, = plt.plot(x_values, genpareto.pdf(x_values, c0, loc0, scale0), 'r-', lw=2, label='GenPareto PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Generalized Pareto Distribution')
plt.legend()

# Add sliders for the parameters
axcolor = 'lightgoldenrodyellow'
ax_c = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor=axcolor)
ax_loc = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=axcolor)
ax_scale = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)

s_c = Slider(ax_c, 'c', 0.01, 5.0, valinit=c0)
s_loc = Slider(ax_loc, 'loc', -5, 5, valinit=loc0)
s_scale = Slider(ax_scale, 'scale', 0.1, 10.0, valinit=scale0)

# Update function to redraw the plot with new parameters
def update(val):
    c = s_c.val
    loc = s_loc.val
    scale = s_scale.val
    pdf_initial.set_ydata(genpareto.pdf(x_values, c, 0, scale))
    fig.canvas.draw_idle()

# Attach the update function to the sliders
s_c.on_changed(update)
s_loc.on_changed(update)
s_scale.on_changed(update)

plt.show()
