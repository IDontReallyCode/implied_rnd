import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error
from math import sqrt

# Define the non_linear_SVI000 function
def non_linear_SVI000(x, a0, a1, a2, b0, b1):
    return a0 + a1 * (x - b0) + a2 * np.sqrt(b1 + (x - b0)**2)

# Load fixed data from .npy files
x = np.load(f'SPX_2022-10-12_x.npy')
y = np.load(f'SPX_2022-10-12_y.npy')
# now sort x and put the value into x_fixed
x_fixed = np.sort(x)
# now sort y according to the index of x and put the value into y_fixed
y_fixed = y[np.argsort(x)]

# Create the main window
root = Tk()
root.title("SVI Plotter")

# Create input labels and entry boxes for parameters
labels = ["a0:", "a1:", "a2:", "b0:", "b1:"]
entries = []

for label in labels:
    Label(root, text=label).pack()
    entry = Entry(root)
    entry.pack()
    entries.append(entry)

# Create a frame for the plot
plot_frame = Frame(root)
plot_frame.pack()

# Function to plot the graph
def plot_graph(a0, a1, a2, b0, b1):
    # Calculate y values for the user-defined function
    y_user = non_linear_SVI000(x_fixed, a0, a1, a2, b0, b1)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_fixed, y_user))

    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the functions
    ax.plot(x_fixed, y_fixed, label="Fixed Data")
    ax.plot(x_fixed, y_user, label="User Parameters")
    ax.set_title(f"SVI Functions (RMSE: {rmse:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to update the graph with user inputs
def update_graph():
    # Get user input parameters
    try:
        a0, a1, a2, b0, b1 = [float(entry.get()) for entry in entries]
    except ValueError:
        print("Please enter valid numeric values.")
        return
    
    # Plot the graph with new parameters
    plot_graph(a0, a1, a2, b0, b1)

# Create a button to update the graph
Button(root, text="Update Graph", command=update_graph).pack()

# Initial plot with fixed parameters and default user parameters
plot_graph(0.15, 0, 0.25, -0.5, 0.1)

# Start the main event loop
root.mainloop()
