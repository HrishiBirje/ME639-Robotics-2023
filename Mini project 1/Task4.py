import numpy as np
import matplotlib.pyplot as plt

L1 = int(input('Length of link 1: '))
L2 = int(input('Length of link 2: '))

# Joint angle limits
theta_min = 35
theta_max = 145

# Converting degree to radians
theta_min_rad = np.radians(theta_min)
theta_max_rad = np.radians(theta_max)

# Range of angles
theta = np.linspace(theta_min_rad, theta_max_rad, 100)

# Initialize arrays to store the end effector positions (x, y)
end_effector_x = []
end_effector_y = []

# Calculate the workspace points
for theta1 in theta:
    for theta2 in theta:
        # Forward kinematics
        x = L1 * np.cos(theta1) + L2 * np.cos(theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta2)

        # End effector data points
        end_effector_x.append(x)
        end_effector_y.append(y)

# Scatter plot of the workspace
plt.scatter(end_effector_x, end_effector_y, s=5)
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Workspace")
plt.ylim(0, L1+L2+1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
