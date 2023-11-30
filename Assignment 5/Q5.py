import numpy as np

def euler_angles_from_U(U):

    u13, u23, u31, u32, u33, u11, u21 = U[0, 2], U[1, 2], U[2, 0], U[2, 1], U[2, 2], U[0,0], U[1,0]

    if u13 ==0 and u23 ==0:
        phi_psi = np.arctan2(u11, u21)

        return phi_psi
    else:

        theta = np.arctan2(u33, np.sqrt(1 - u33 ** 2))
        phi = np.arctan2(u13, u23)
        psi = np.arctan2(-u31, u32)

        return theta, phi, psi