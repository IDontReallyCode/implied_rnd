import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error
from math import sqrt

# Define the non_linear_SVI000 function
def non_linear_SVI000(x, a0, a1, a2, b0, b1):
    return a0 + a1 * (x - b0) + a2 * np.sqrt(b1 + (x - b0)**2)

# Fixed parameters for the first function
a0_fixed, a1_fixed, a2_fixed, b0_fixed, b1_fixed = 1, 1, 1, 1, 1

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
    # Generate x values
    x = np.linspace(-5, 2, 500)

    # Calculate y values for both functions
    y_fixed = non_linear_SVI000(x, a0_fixed, a1_fixed, a2_fixed, b0_fixed, b1_fixed)
    y_user = non_linear_SVI000(x, a0, a1, a2, b0, b1)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_fixed, y_user))

    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the functions
    ax.plot(x, y_fixed, label="Fixed Parameters")
    ax.plot(x, y_user, label="User Parameters")
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
plot_graph(a0_fixed, a1_fixed, a2_fixed, b0_fixed, b1_fixed)

# Start the main event loop
root.mainloop()
