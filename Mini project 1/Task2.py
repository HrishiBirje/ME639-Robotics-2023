import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Arm lengths
# L1 = float(input('Length of link 1: '))
# L2 = float(input('Length of link 2: '))
L1 = 1
L2 = 1

# Coordinates
# x = float(input('x: '))
# y = float(input('y: '))
x = 1
y = 0.75
# N = float(input('Force: '))
N = 5
r = np.sqrt(x**2 + y**2)

if (L1 + L2) < r:
    print('Out of workspace')

else:

    # Wall
    x1 = 2
    x2 = 0
    y1 = 0.75
    y2 = 0.75
    #p, q = [x1, x2], [y1, y2]

    # Finding angles
    theta = np.arccos((x**2 + y**2 - L1**2 - L2**2)/(2*L1*L2))
    q1 = np.arctan(y/x) - np.arctan(L2*np.sin(theta)/(L1 + L2*np.cos(theta)))
    q2 = q1 + theta

    # Finding torque
    slope = np.arctan((y2 - y1)/(x2 - x1))
    Nx = N*np.sin(slope)
    Ny = N*np.cos(slope)
    a = np.array([[-L1*np.sin(q1), L1*np.cos(q1)], [-L2*np.sin(q2), L2*np.cos(q2)]])
    b = np.array([Nx, Ny])
    c = np.matmul(a, b)
    T1 = c[0]
    T2 = c[1]
    print('Torque 1 = ', T1)
    print('Torque 2 = ', T2)

    # Animation
    num_steps = 10
    theta1_range = np.linspace(0.1, q1, num_steps)
    theta2_range = np.linspace(0.1, q2, num_steps)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-', lw=2)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    #ax.plot(p, q, label='wall')  # Plot the wall line
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2R Robotic Arm Simulation')

    # Initialization function for animation
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        theta1 = theta1_range[i]
        theta2 = theta2_range[i]
        m = [0, L1 * np.cos(theta1), L1 * np.cos(theta1) + L2 * np.cos(theta2)]
        n = [0, L1 * np.sin(theta1), L1 * np.sin(theta1) + L2 * np.sin(theta2)]
        line.set_data(m, n)

        return line,

    ani = FuncAnimation(fig, animate, frames=num_steps, init_func=init, blit=True)

    plt.show()

