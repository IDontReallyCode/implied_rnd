import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def update_plot(a0, a1, b0, a2, b1):
    x = np.linspace(-2, 3, 100)
    y = a0 + a1 * (x - b0) + a2 * np.sqrt(b1 + (x - b0)**2) - a2 * np.sqrt(b1)
    ax.clear()
    ax.plot(x, y)
    canvas.draw()

def slider_update(val):
    a0 = slider_a0.get()
    a1 = slider_a1.get()
    b0 = slider_b0.get()
    a2 = slider_a2.get()
    b1 = slider_b1.get()
    update_plot(a0, a1, b0, a2, b1)

root = tk.Tk()
root.title("Function Plotter")

# Create sliders
sliders = [
    ('a0', 0, 1, 0),
    ('a1', -4, +4, 0),
    ('b0', -2, +2, 0),
    ('a2', 0, +2, 1),
    ('b1', 0, +1, 1)
]

for i, (label, min_val, max_val, init_val) in enumerate(sliders):
    tk.Label(root, text=label).grid(row=i, column=0)
    scale = tk.Scale(root, from_=min_val, to=max_val, orient=tk.HORIZONTAL, resolution=0.1, command=slider_update)
    scale.set(init_val)
    scale.grid(row=i, column=1)

slider_a0 = root.children['!scale']
slider_a1 = root.children['!scale2']
slider_b0 = root.children['!scale3']
slider_a2 = root.children['!scale4']
slider_b1 = root.children['!scale5']

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)

update_plot(1, 1, 1, 1, 1)
root.mainloop()
