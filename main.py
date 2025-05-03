import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical Constants
k = 8.99e9  # Coulomb's constant in N·m²/C²
e = 1.6e-19  # Elementary charge in C
m = 9.11e-31  # Electron mass in kg

# Define the system of differential equations
def electron_dynamics(t, y):
    # Unpack variables
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = y
    
    # Compute distance
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    if r < 1e-10:
        r = 1e-10

    # Compute acceleration due to Coulomb force
    force = k * e**2 / (m * r**2)  # F = k * q1 * q2 / r², mass m included
    ax1 = force * dx / r
    ay1 = force * dy / r
    ax2 = -ax1
    ay2 = -ay1

    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]

# Initial Conditions (to be fine-tuned)
x1_0, y1_0 = -1.75, 0  # Electron 1 position
x2_0, y2_0 = 1.75, 0   # Electron 2 position
vx1_0, vy1_0 = 0, 0.1  # Electron 1 velocity
vx2_0, vy2_0 = 0, -0.1  # Electron 2 velocity

# Solve the system
t_span = (0, 1e-11)  # Time span
y0 = [x1_0, y1_0, vx1_0, vy1_0, x2_0, y2_0, vx2_0, vy2_0]  # Initial state
sol = solve_ivp(electron_dynamics, t_span, y0, method='RK45', t_eval=np.linspace(*t_span, 5000))

# Extract solutions
x1, y1, x2, y2 = sol.y[0], sol.y[1], sol.y[4], sol.y[5]

# Plot the paths of the electrons
plt.figure(figsize=(8, 8))
plt.plot(x1, y1, label="Electron 1", color='blue')
plt.plot(x2, y2, label="Electron 2", color='red')
plt.scatter([x1[0], x2[0]], [y1[0], y2[0]], color='green', label="Start Positions")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Electron Interaction Simulation")
plt.legend()
plt.grid()
plt.show()
