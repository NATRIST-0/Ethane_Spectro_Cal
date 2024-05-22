
"""
Calibration of the Big Spectrophotometer + Sliders + Data Box
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from matplotlib.widgets import TextBox

# Read in the text file
data = pd.read_csv(r"C:\Users\GAYRARD\Desktop\Python\Ethane_Spectro_Calibration\240419_000001.txt", sep=' ')


# Extracting data columns
Date = data.iloc[0, 1]
Ethane = data.iloc[:, 1]

# Time in days
Time_Days = ( data.iloc[:, 0] / 86400 )


#Conversion to UTC
Time = pd.to_datetime(Time_Days , unit = ('D'), origin = Date )

# Create figure and axes
fig, axes = plt.subplots(nrows=1,ncols=2)
plt.subplots_adjust(bottom=0.4)

# Define initial range and plot it
initial_range = (0, 30)
line, = axes[0].plot(Time, Ethane, color='blue', alpha=0.5)
highlight1, = axes[0].plot(Time[initial_range[0]:initial_range[1]], Ethane[initial_range[0]:initial_range[1]], 'o', color='red', markersize=5)
highlight2, = axes[0].plot(Time[initial_range[0]:initial_range[1]], Ethane[initial_range[0]:initial_range[1]], 'o', color='green', markersize=5)
highlight3, = axes[0].plot(Time[initial_range[0]:initial_range[1]], Ethane[initial_range[0]:initial_range[1]], 'o', color='orange', markersize=5)
highlight4, = axes[0].plot(Time[initial_range[0]:initial_range[1]], Ethane[initial_range[0]:initial_range[1]], 'o', color='purple', markersize=5)


# Déclarer les valeurs en dehors de la fonction
val1 = initial_range[0]
val2 = initial_range[0]
val3 = initial_range[0]
val4 = initial_range[0]

# Définir la fonction pour mettre à jour les valeurs et le tracé
def update(val1, val2, val3, val4):
    start1 = int(val1)
    end1 = start1 + 30
    start2 = int(val2)
    end2 = start2 + 30
    start3 = int(val3)
    end3 = start3 + 30
    start4 = int(val4)
    end4 = start4 + 30

    # Obtenir les valeurs de l'axe Y correspondantes
    y_values1 = Ethane[start1:end1].values
    y_values2 = Ethane[start2:end2].values
    y_values3 = Ethane[start3:end3].values
    y_values4 = Ethane[start4:end4].values

    # Mettre à jour les données pour chaque série de points de mise en évidence
    highlight1.set_xdata(Time[start1:end1])
    highlight1.set_ydata(y_values1)
    
    highlight2.set_xdata(Time[start2:end2])
    highlight2.set_ydata(y_values2)
    
    highlight3.set_xdata(Time[start3:end3])
    highlight3.set_ydata(y_values3)
    
    highlight4.set_xdata(Time[start4:end4])
    highlight4.set_ydata(y_values4)

    # Afficher les valeurs de l'axe Y
    # print("Valeurs de l'axe Y pour le curseur 1:", y_values1)
    # print("Valeurs de l'axe Y pour le curseur 2:", y_values2)
    # print("Valeurs de l'axe Y pour le curseur 3:", y_values3)
    # print("Valeurs de l'axe Y pour le curseur 4:", y_values4)

    fig.canvas.draw_idle()

def update_sliders(val):
    update(slider1.val, slider2.val, slider3.val, slider4.val)
    # Update the text displayed to the left of the sliders
    slider1.valtext.set_text(f'{Ethane[int(slider1.val)]:.2f}')
    slider2.valtext.set_text(f'{Ethane[int(slider2.val)]:.2f}')
    slider3.valtext.set_text(f'{Ethane[int(slider3.val)]:.2f}')
    slider4.valtext.set_text(f'{Ethane[int(slider4.val)]:.2f}')  
    

# Create sliders with colors matching the highlights
ax_slider1 = plt.axes([0.085, 0.25, 0.4, 0.03], facecolor='lightgoldenrodyellow')
slider1 = Slider(ax_slider1, 'Ethane Zero\n(ppbv)', 0, len(Time) - 30, valinit=initial_range[0], color='red')

ax_slider2 = plt.axes([0.085, 0.20, 0.4, 0.03], facecolor='lightgoldenrodyellow')
slider2 = Slider(ax_slider2, 'Ethane 1st Dilution\n(ppbv)', 0, len(Time) - 30, valinit=initial_range[0], color='green')

ax_slider3 = plt.axes([0.085, 0.15, 0.4, 0.03], facecolor='lightgoldenrodyellow')
slider3 = Slider(ax_slider3, 'Ethane 2nd Dilution\n(ppbv)', 0, len(Time) - 30, valinit=initial_range[0], color='orange')

ax_slider4 = plt.axes([0.085, 0.10, 0.4, 0.03], facecolor='lightgoldenrodyellow')
slider4 = Slider(ax_slider4, 'Ethane 3rd Dilution\n(ppbv)', 0, len(Time) - 30, valinit=initial_range[0], color='purple')


# Link the callback function to the sliders
slider1.on_changed(update_sliders)
slider2.on_changed(update_sliders)
slider3.on_changed(update_sliders)
slider4.on_changed(update_sliders)

# Set labels and title
axes[0].set_xlabel('Date and Time')
axes[0].set_ylabel('Ethane Concentration (ppbv)')
axes[0].set_title(f'{Time.name} Ethane Calibration Raw Data')
axes[0].grid(True)

plt.show()

# Set up the TextBox widgets
zero_box = TextBox(plt.gcf().add_axes([0.65, 0.25, 0.15, 0.03]), 'True 0 (ppbv) =')
dil1_box = TextBox(plt.gcf().add_axes([0.65, 0.20, 0.15, 0.03]), 'Dil 1 Concentration (ppbv) =')
dil2_box = TextBox(plt.gcf().add_axes([0.65, 0.15, 0.15, 0.03]), 'Dil 2 Concentration (ppbv) =')
dil3_box = TextBox(plt.gcf().add_axes([0.65, 0.10, 0.15, 0.03]), 'Dil 3 Concentration (ppbv) =')

# Define the submit function
def submit(val):
    # Get the values from TextBox widgets
    zero = float(zero_box.text)
    dil1 = float(dil1_box.text)
    dil2 = float(dil2_box.text)
    dil3 = float(dil3_box.text)
    
    # Compute average values from sliders
    avg1 = np.mean(Ethane[int(slider1.val):int(slider1.val) + 30])
    avg2 = np.mean(Ethane[int(slider2.val):int(slider2.val) + 30])
    avg3 = np.mean(Ethane[int(slider3.val):int(slider3.val) + 30])
    avg4 = np.mean(Ethane[int(slider4.val):int(slider4.val) + 30])
    
    # Perform linear regression
    x = np.array([zero, dil1, dil2, dil3])
    y = np.array([avg1, avg2, avg3, avg4])
    coeffs = np.polyfit(x, y, 1)
    line = np.polyval(coeffs, x)
    
    # Calculate R squared
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    # Clear the second axes
    axes[1].clear()
    
    # Plot the average values against concentrations
    axes[1].plot(x, y, 'o', label='Data')
    
    # Plot the regression line
    axes[1].plot(x, line, label='Linear Regression', color='red')
    
    # Display the equation of the line and R squared
    equation = f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}'
    r_squared_text = f'R² = {r_squared:.2f}'
    axes[1].text(0.05, 0.95, equation, transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    axes[1].text(0.05, 0.90, r_squared_text, transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    
    # Set labels and title for the second plot
    axes[1].set_xlabel('Theoretical Ethane Concentration (ppbv)')
    axes[1].set_ylabel('Mesured Ethane Concentration (ppbv)')
    axes[1].set_title(f'Ethane Calibration from {Time.name}')
    
    # Draw the plot
    plt.draw()

# Attach the submit function to the TextBox widgets
zero_box.on_submit(submit)
dil1_box.on_submit(submit)
dil2_box.on_submit(submit)
dil3_box.on_submit(submit)

plt.show()