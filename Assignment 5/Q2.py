import numpy as np

def joint_vel(J, X_dot):

    J_inv = np.linalg.pinv(J) #Inverse
    q_dot = np.dot(J_inv, X_dot)

    return q_dot

