import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import genpareto

# Load the numpy arrays
# tailx = np.load('tailx.npy')
# tailf = np.load('tailf.npy')
tailx = np.array([1, 2])
tailf = np.array([0.2, 0.1])

# Scatter plot of tailx and tailf
fig, ax = plt.subplots()
plt.scatter(tailx, tailf, label='Data Points')
plt.xlabel('tailx')
plt.ylabel('tailf')
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

s_c = Slider(ax_c, 'c', 0.01, 20.0, valinit=c0)
s_loc = Slider(ax_loc, 'loc', -5, 0, valinit=loc0)
s_scale = Slider(ax_scale, 'scale', 0.1, 20.0, valinit=scale0)

# Update function to redraw the plot with new parameters and adjust y-axis limits
def update(val):
    c = s_c.val
    loc = s_loc.val
    scale = s_scale.val
    new_pdf = genpareto.pdf(x_values, c, loc, scale)
    pdf_initial.set_ydata(new_pdf)
    ax.set_ylim([0, max(new_pdf) * 1.1])  # Adjust the y-axis limit dynamically
    fig.canvas.draw_idle()

# Attach the update function to the sliders
s_c.on_changed(update)
s_loc.on_changed(update)
s_scale.on_changed(update)

plt.show()
