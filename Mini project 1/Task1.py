import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Make a graph
# Vary Angles
# Join the lines

# Arm lengths
l1 = float(input('Length of link 1: '))
l2 = float(input('Length of link 2: '))

# Time array
num_steps = 10
t = np.linspace(0.1, 1.5, num_steps)


#Animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
line, = ax.plot([], [], 'o-', lw = 2, markersize = 10)

def init():
    line.set_data([], [])
    return line,

def animate(i):

    x = t[i]
    y = np.cos(t[i])

    theta = np.arccos((x**2 + y**2 - l1**2 - l2**2)/2*l1*l2)
    q1 = np.arctan(y/x) - np.arctan(l2*np.sin(theta)/(l1 + l2*np.cos(theta)))
    q2 = q1 + theta

    a = [0, l1*np.cos(q1), l1*np.cos(q1) + l2*np.cos(q2)]
    b = [0, l1*np.sin(q1), l1*np.sin(q1) + l2*np.sin(q2)]
    line.set_data(a, b)
    return line,

ani = FuncAnimation(fig, animate, frames = num_steps, init_func = init, blit = True)

plt.show()