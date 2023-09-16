import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define system parameters
m1 = 1.0    # mass of the first link (kg)
m2 = 1.0    # mass of the second link (kg)
l1 = 1.0    # length of the first link (m)
l2 = 1.0    # length of the second link (m)
k = 0.2    # spring constant

# Define the Lagrange's equations of motion with spring force
def lagrange_equations(state, t):
    theta1, theta2, theta1_dot, theta2_dot = state

    # Inertia tensors
    I1 = (1/3) * m1 * l1**2
    I2 = (1/3) * m2 * l2**2

    # Mass matrix
    M11 = (m1 + m2) * l1*2 + m2 * l2*2 + 2 * m2 * l1 * l2 * np.cos(theta2)
    M12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(theta2)
    M21 = M12
    M22 = m2 * l2**2

    # Coriolis and centrifugal terms
    c1 = -m2 * l1 * l2 * theta2_dot * (2 * theta1_dot + theta2_dot) * np.sin(theta2)
    c2 = m2 * l1 * l2 * theta1_dot**2 * np.sin(theta2)

    # Spring force components
    spring_force_x = -k * (x2 - x_s)
    spring_force_y = -k * (y2 - y_s)

    # Calculate torques as a result of spring force
    torque1 = - spring_force_x * l1 * np.sin(theta1) + spring_force_y * l1 * np.cos(theta1)
    torque2 = - spring_force_x * l2 * np.sin(theta1 + theta2) + spring_force_y * l2 * np.cos(theta1 + theta2)

    # Equations of motion
    theta1_ddot = (M22 * torque1 - M12 * torque2 - c1 - c2) / (M11 * M22 - M12 * M21)
    theta2_ddot = (M11 * torque2 - M21 * torque1 + c1) / (M11 * M22 - M12 * M21)

    return [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]

# Initial conditions
initial_state = [0.0, 0.0, 0.0, 0.0]  # theta1, theta2, theta1_dot, theta2_dot

# Time array
t_max = 10.0
num_steps = 250
time = np.linspace(0, t_max, num_steps)

# Arbitrary points for spring attachment and initial end effector position
x_s, y_s = 1, 1
x2, y2 = 2, 0  # Initial position of the end effector

# Solve the equations of motion using odeint
solutions = odeint(lagrange_equations, initial_state, time)

# Extract solutions for theta1, theta2, theta1_dot, theta2_dot
theta1_solutions = solutions[:, 0]
theta2_solutions = solutions[:, 1]
theta1_dot_solutions = solutions[:, 2]
theta2_dot_solutions = solutions[:, 3]

# Animation function
def animate(i):
    theta1 = theta1_solutions[i]
    theta2 = theta2_solutions[i]

    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    spring_x = [x2, x_s]
    spring_y = [y2, y_s]

    link1.set_data([0, x1], [0, y1])
    link2.set_data([x1, x2], [y1, y2])
    spring_line.set_data(spring_x, spring_y)

    return link1, link2, spring_line

# Create animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
link1, = ax.plot([], [], 'b-', lw=4)
link2, = ax.plot([], [], 'r-', lw=4)
spring_line, = ax.plot([], [], 'g-', lw=2)

ani = FuncAnimation(fig, animate, frames=num_steps, blit=True, interval=(t_max / num_steps) * 1000)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2R Manipulator Animation with Attached Spring Torques')
plt.grid()

from IPython.display import HTML
HTML(ani.to_jshtml())