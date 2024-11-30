# In this test program, I want to manually set the theta parameter for the Generalized Pareto distribution.
# then plot what the distribution looks like.
# I want to use the visualization tool where I can have sliders to change the parameters and see the distribution change.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GenParetoVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Generalized Pareto Distribution Visualizer")

        self.shape = tk.DoubleVar(value=1.0)
        self.loc = tk.DoubleVar(value=0.0)
        self.scale = tk.DoubleVar(value=1.0)

        self.create_widgets()
        self.plot_distribution()

    def create_widgets(self):
        tk.Label(self.root, text="Shape (c)").pack()
        tk.Scale(self.root, variable=self.shape, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, command=self.update_plot).pack()

        tk.Label(self.root, text="Location (loc)").pack()
        tk.Scale(self.root, variable=self.loc, from_=-10.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_plot).pack()

        tk.Label(self.root, text="Scale (scale)").pack()
        tk.Scale(self.root, variable=self.scale, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_plot).pack()

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

    def plot_distribution(self):
        x = np.linspace(-10, 10, 1000)
        shape = self.shape.get()
        loc = self.loc.get()
        scale = self.scale.get()

        density = genextreme(shape, loc, scale)
        self.ax.clear()
        self.ax.plot(x, density.pdf(x), label=f'c={shape}, loc={loc}, scale={scale}')
        self.ax.set_title("Generalized Pareto Distribution")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("PDF")
        self.ax.legend()
        self.canvas.draw()

    def update_plot(self, event=None):
        self.plot_distribution()

if __name__ == "__main__":
    root = tk.Tk()
    app = GenParetoVisualizer(root)
    root.mainloop()

