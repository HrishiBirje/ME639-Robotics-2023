import numpy as np

x = float(input("x: "))
y = float(input("y: "))
z = float(input("z: "))
l1 = float(input("Link l1: "))
l2 = float(input("Link l2: "))
l3 = float(input("Link l3: "))

def getH(R, d):
    H=[[R[0][0],R[0][1], R[0][2],d[0][0]],
       [R[1][0],R[1][1], R[1][2],d[1][0]],
       [R[2][0],R[2][1], R[2][2],d[2][0]],
       [0,0, 0,1]]
    return H

def getR(q):
    R=[[np.cos(q), -np.sin(q), 0],
       [np.sin(q), np.cos(q), 0],
       [0, 0, 1]]
    return R

q1 = np.arctan2(x,y)
q2 = np.arctan2(z-l1, np.sqrt(x**2+y**2))
d = np.sqrt(x**2 + y**2 + (z-l1)**2)-(l2+l3)

H01 = getH(getR(q1), [[0],[0],[0]])
H12 = getH(np.matmul([[1, 0, 0],[0, 0, -1],[0, 1, 0]], getR(q2)), [[0],[0],[l1]])
H23 = getH(getR(0), [[l2+l3],[0],[0]])
P = np.linalg.multi_dot([H01,H12,H23,[[d],[0],[0],[1]]])

print(f"position: {P[0][0]}i {P[1][0]}j {P[2][0]}k")


