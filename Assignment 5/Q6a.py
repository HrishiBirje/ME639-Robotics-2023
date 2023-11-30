import numpy as np
import matplotlib.pyplot as plt

m1 = float(input('m1 = '))
m2 = float(input('m2 = '))
m3 = float(input('m3 = '))

l1 = float(input('l1 = '))
l2 = float(input('l2 = '))
l3 = float(input('l3 = '))

I1 = float(input('I1 = '))
I2 = float(input('I2 = '))
I3 = float(input('I3 = '))

# Gravity constant
g = 9.81

dt = 0.01  # Time step
t_max = 10.0  # Maximum simulation time
num_steps = int(t_max / dt)  # Number of time steps

# Initial conditions
q1_0 = 0.1
q2_0 = 0.1
q3_0 = 0.1
dq1_0 = 0.0
dq2_0 = 0.0
dq3_0 = 0.0

# Constant torques
tau1 = 0.01
tau2 = 0.01
tau3 = 0.01

# Arrays to store simulation results
time = np.linspace(0, t_max, num_steps)
q1 = np.zeros(num_steps)
q2 = np.zeros(num_steps)
q3 = np.zeros(num_steps)

# Initialize state variables
q1[0] = q1_0
q2[0] = q2_0
q3[0] = q3_0
dq1 = dq1_0
dq2 = dq2_0
dq3 = dq3_0

# Perform numerical integration using Euler's method
for i in range(1, num_steps):
    # Compute accelerations using the dynamic equations
    ddq1 = (tau1 - m1 * g * l1 * np.sin(q1[i-1]) - m2 * g * (l1 * np.sin(q1[i-1]) + l2 * np.sin(q1[i-1] + q2[i-1])) - m3 * g * (l1 * np.sin(q1[i-1]) + l2 * np.sin(q1[i-1] + q2[i-1]) + l3 * np.sin(q1[i-1] + q2[i-1] + q3[i-1]))) / (I1 + I2 + I3)
    ddq2 = (tau2 - m2 * g * l2 * np.sin(q1[i-1] + q2[i-1]) - m3 * g * (l2 * np.sin(q1[i-1] + q2[i-1]) + l3 * np.sin(q1[i-1] + q2[i-1] + q3[i-1]))) / (I2 + I3)
    ddq3 = (tau3 - m3 * g * l3 * np.sin(q1[i-1] + q2[i-1] + q3[i-1])) / I3

    # Update velocities and positions
    dq1 += ddq1 * dt
    dq2 += ddq2 * dt
    dq3 += ddq3 * dt
    q1[i] = q1[i-1] + dq1 * dt
    q2[i] = q2[i-1] + dq2 * dt
    q3[i] = q3[i-1] + dq3 * dt

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(time, q1)
plt.title('Joint 1 Angle')
plt.subplot(312)
plt.plot(time, q2)
plt.title('Joint 2 Angle')
plt.subplot(313)
plt.plot(time, q3)
plt.title('Joint 3 Angle')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
