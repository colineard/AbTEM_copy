# import numpy as np
# import matplotlib.pyplot as plt
# from math import comb, factorial # CHANGED: Imported from the standard 'math' library

# p = 4                     # Degree of LG mode
# l = 5                     # Order of LG mode
# w0 = 2.0                  # Beam waist
# k = 2 * np.pi / 532.0e-9  # Wavenumber of light (532 nm, green laser)

# zR = k * w0**2 / 2        # Calculate the Rayleigh range

# # Setup the cartesian grid for the plot at plane z
# z = 0.0
# x_vec = np.linspace(-5, 5, 500) # Increased resolution for a smoother plot
# y_vec = np.linspace(-5, 5, 500)
# xx, yy = np.meshgrid(x_vec, y_vec)

# # Calculate the cylindrical coordinates
# r = np.sqrt(xx**2 + yy**2)
# phi = np.arctan2(yy, xx)

# # --- Calculate beam properties ---
# U00 = 1.0 / (1 + 1j * z / zR) * np.exp(-r**2 / w0**2 / (1 + 1j * z / zR))
# w = w0 * np.sqrt(1.0 + z**2 / zR**2)
# R = np.sqrt(2.0) * r / w

# # --- Calculate Laguerre Polynomial Lpl ---
# # Lpl formula from OT toolbox (Nieminen et al., 2004)
# Lpl = np.zeros_like(R)
# for m in range(p + 1): # CHANGED: Used 'range' instead of 'xrange'
#     Lpl += ((-1.0)**m / factorial(m)) * comb(p + abs(l), p - m) * R**(2.0 * m)

# # --- Calculate the final complex field U ---
# gouy_phase = np.arctan(z / zR)
# U = (
#     U00
#     * R**abs(l)
#     * Lpl
#     * np.exp(1j * l * phi)
#     * np.exp(-1j * (2 * p + abs(l) + 1) * gouy_phase)
# )

# # --- Plotting --- 🎨
# # Define the plot extent for imshow
# plot_extent = [x_vec.min(), x_vec.max(), y_vec.min(), y_vec.max()]

# # Plot Intensity
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title(f'Intensity (p={p}, l={l})')
# plt.imshow(np.abs(U)**2, extent=plot_extent, cmap='inferno')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()

# # Plot Phase
# plt.subplot(1, 2, 2)
# plt.title(f'Phase (p={p}, l={l})')
# # CHANGED: Plotted np.angle(U) directly with a cyclic colormap
# plt.imshow(np.angle(U), extent=plot_extent, cmap='hsv')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.colorbar()

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from math import comb, factorial

p = 4                     # Degree of LG mode
l = 5                     # Order of LG mode
w0 = 2.0                  # Beam waist
k = 2 * np.pi / 532.0e-9  # Wavenumber of light (532 nm, green laser)

zR = k * w0**2 / 2        # Calculate the Rayleigh range

# Setup the cartesian grid for the plot at plane z
z = 0.0
x_vec = np.linspace(-5, 5, 500)
y_vec = np.linspace(-5, 5, 500)
xx, yy = np.meshgrid(x_vec, y_vec)

# Calculate the cylindrical coordinates
r = np.sqrt(xx**2 + yy**2)
phi = np.arctan2(yy, xx)

# --- Calculate beam properties ---
U00 = 1.0 / (1 + 1j * z / zR) * np.exp(-r**2 / w0**2 / (1 + 1j * z / zR))
w = w0 * np.sqrt(1.0 + z**2 / zR**2)
R = np.sqrt(2.0) * r / w

# --- Calculate Laguerre Polynomial Lpl ---
Lpl = np.zeros_like(R)
for m in range(p + 1):
    Lpl += ((-1.0)**m / factorial(m)) * comb(p + abs(l), p - m) * R**(2.0 * m)

# --- Calculate the final complex field U ---
gouy_phase = np.arctan(z / zR)
U = (
    U00
    * R**abs(l)
    * Lpl
    * np.exp(1j * l * phi)
    * np.exp(-1j * (2 * p + abs(l) + 1) * gouy_phase)
)

# --- Plotting Real Space --- 🎨
plot_extent = [x_vec.min(), x_vec.max(), y_vec.min(), y_vec.max()]

plt.figure(figsize=(10, 5))
plt.suptitle('Real Space', fontsize=16)

plt.subplot(1, 2, 1)
plt.title(f'Intensity (p={p}, l={l})')
plt.imshow(np.abs(U)**2, extent=plot_extent, cmap='inferno', norm=plt.matplotlib.colors.LogNorm())
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title(f'Phase (p={p}, l={l})')
plt.imshow(np.angle(U), extent=plot_extent, cmap='hsv')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

plt.tight_layout(rect=[0, 0, 1, 0.96])


# --- [ADDED] Calculate and Plot Fourier Transform --- 🛰️

# 1. Perform the 2D Fourier Transform
# The ifftshift/fftshift pattern ensures the zero-frequency is centered
U_k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))

# 2. Create the reciprocal space coordinate axes (kx, ky)
# The frequency bins are determined by the real-space sampling
dx = x_vec[1] - x_vec[0]
k_axis = np.fft.fftshift(np.fft.fftfreq(len(x_vec), d=dx))
plot_extent_k = [k_axis.min(), k_axis.max(), k_axis.min(), k_axis.max()]

# 3. Plot the Reciprocal Space Intensity and Phase
plt.figure(figsize=(10, 5))
plt.suptitle('Reciprocal Space (Fourier Transform)', fontsize=16)

# Plot Reciprocal Space Intensity
plt.subplot(1, 2, 1)
plt.title(f'Intensity (p={p}, l={l})')
plt.imshow(np.abs(U_k)**2, extent=plot_extent_k, cmap='inferno', norm=plt.matplotlib.colors.LogNorm())
plt.xlabel('kx')
plt.ylabel('ky')
plt.colorbar()

# Plot Reciprocal Space Phase
plt.subplot(1, 2, 2)
plt.title(f'Phase (p={p}, l={l})')
plt.imshow(np.angle(U_k), extent=plot_extent_k, cmap='hsv')
plt.xlabel('kx')
plt.ylabel('ky')
plt.colorbar()

plt.tight_layout(rect=[0, 0, 1, 0.96])

# --- Show all plots ---
plt.show()
