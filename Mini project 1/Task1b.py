import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define system parameters
m1 = 1.0    # mass of the first link (kg)
m2 = 1.0    # mass of the second link (kg)
l1 = 1.0    # length of the first link (m)
l2 = 1.0    # length of the second link (m)

x = float(input('x = '))
y = float(input('y = '))

theta = np.arccos((x**2 + y**2 - l1**2 - l2**2)/2*l1*l2)
q1 = np.arctan2(y/x) - np.arctan2(l2*np.sin(theta)/(l1 + l2*np.cos(theta)))
q2 = q1 + theta
a = [0, l1*np.cos(q1), l1*np.cos(q1) + l2*np.cos(q2)]
b = [0, l1*np.sin(q1), l1*np.sin(q1) + l2*np.sin(q2)]


# Define the Lagrange's equations of motion
def lagrange_equations(state, t, torque1, torque2):
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

    # Equations of motion
    theta1_ddot = (M22 * (torque1 - c1 - c2) - M12 * (torque2 + c1)) / (M11 * M22 - M12 * M21)
    theta2_ddot = (M11 * (torque2 + c1) - M21 * (torque1 - c1 - c2)) / (M11 * M22 - M12 * M21)

    return [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]

# PD control gains
kp = 1.0  # Proportional gain
kd = 1.0   # Derivative gain

# PD control function to calculate torques
def pd_control(current_theta1, current_theta2, current_theta1_dot, current_theta2_dot, desired_theta1, desired_theta2):
    error_theta1 = desired_theta1 - current_theta1
    error_theta2 = desired_theta2 - current_theta2
    torque1 = kp * error_theta1 - kd * current_theta1_dot
    torque2 = kp * error_theta2 - kd * current_theta2_dot
    return torque1, torque2

# Initial conditions
initial_state = [0.0, 0.0, 0.0, 0.0]  # theta1, theta2, theta1_dot, theta2_dot

# Time array
t_max = 40.0
num_steps = 400
time = np.linspace(0, t_max, num_steps)

# Animation function
def animate(i):
    global solutions

    current_theta1 = theta1_solutions[i]
    current_theta2 = theta2_solutions[i]

    torque1, torque2 = pd_control(current_theta1, current_theta2, theta1_dot_solutions[i], theta2_dot_solutions[i], desired_theta1, desired_theta2)

    solutions = odeint(lagrange_equations, solutions[-1], [time[i], time[i + 1]], args=(torque1, torque2))

    theta1_solutions[i + 1] = solutions[-1, 0]
    theta2_solutions[i + 1] = solutions[-1, 1]
    theta1_dot_solutions[i + 1] = solutions[-1, 2]
    theta2_dot_solutions[i + 1] = solutions[-1, 3]

    theta1 = theta1_solutions[i + 1]
    theta2 = theta2_solutions[i + 1]

    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    link1.set_data([0, x1], [0, y1])
    link2.set_data([x1, x2], [y1, y2])

    return link1, link2

# Solve the equations of motion using odeint with PD control
solutions = odeint(lagrange_equations, initial_state, time, args=(0.0, 0.0))

# Extract solutions for theta1, theta2, theta1_dot, theta2_dot
theta1_solutions = solutions[:, 0]
theta2_solutions = solutions[:, 1]
theta1_dot_solutions = solutions[:, 2]
theta2_dot_solutions = solutions[:, 3]

# Create animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
link1, = ax.plot([], [], 'b-', lw=4)
link2, = ax.plot([], [], 'r-', lw=4)

ani = FuncAnimation(fig, animate, frames=num_steps - 1, blit=True, interval=(t_max / num_steps) * 1000)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('2R Manipulator Animation')
plt.grid()

from IPython.display import HTML
HTML(ani.to_jshtml())