import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error

# Load the test_x and test_y data
test_x = pd.read_csv('test_x.csv').values.flatten()
test_y = pd.read_csv('test_y.csv').values.flatten()

def update_plot(a0, a1, b0, a2, b1):
    x = np.linspace(min(test_x), max(test_x), 100)
    y = a0 + a1 * (x - b0) + a2 * np.sqrt(b1 + (x - b0)**2) - a2 * np.sqrt(b1)
    ax.clear()
    ax.plot(x, y, label='Fitted Function')
    ax.plot(test_x, test_y, 'ro', label='Test Data')
    
    # Calculate and display RMSE
    rmse = np.sqrt(mean_squared_error(test_y, a0 + a1 * (test_x - b0) + a2 * np.sqrt(b1 + (test_x - b0)**2) - a2 * np.sqrt(b1)))
    ax.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=ax.transAxes, verticalalignment='top')

    ax.legend()
    canvas.draw()

def slider_update(val):
    a0 = slider_a0.get()
    a1 = slider_a1.get()
    b0 = slider_b0.get()
    a2 = slider_a2.get()
    b1 = slider_b1.get()
    update_plot(a0, a1, b0, a2, b1)

def create_slider_with_spinbox(root, label, min_val, max_val, init_val, row, column, resolution, command):
    tk.Label(root, text=label).grid(row=row, column=column)
    scale = tk.Scale(root, from_=min_val, to=max_val, orient=tk.HORIZONTAL, resolution=resolution, length=600, command=command)
    scale.set(init_val)
    scale.grid(row=row, column=column + 1)
    spinbox = tk.Spinbox(root, from_=min_val, to=max_val, increment=resolution, command=command, width=5)
    spinbox.delete(0, "end")
    spinbox.insert(0, init_val)
    spinbox.grid(row=row, column=column + 2)
    return scale, spinbox

root = tk.Tk()
root.title("Function Plotter")

# Create sliders with spinboxes
sliders = [
    ('a0', 0, 1, 0),
    ('a1', -4, +4, 0),
    ('b0', -2, +2, 0),
    ('a2', 0, +2, 1),
    ('b1', 0, +1, 1)
]

scales = []
spinboxes = []
for i, (label, min_val, max_val, init_val) in enumerate(sliders):
    scale, spinbox = create_slider_with_spinbox(root, label, min_val, max_val, init_val, i, 0, 0.01, slider_update)
    scales.append(scale)
    spinboxes.append(spinbox)

slider_a0 = scales[0]
slider_a1 = scales[1]
slider_b0 = scales[2]
slider_a2 = scales[3]
slider_b1 = scales[4]

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=6, column=0, columnspan=3)

update_plot(1, 1, 1, 1, 1)
root.mainloop()
