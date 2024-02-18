import osqp
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Problem visualization
h = 1e-2
X, Y = np.meshgrid(np.arange(-1, 1 + h, h), np.arange(-1, 1 + h, h))
Z = X**2 + Y**2
    
plt.figure()
plt.axis("equal")
plt.contour(X, Y, Z, levels=20)
plt.plot([0, 1], [-1, 1], color="red")

# Problem definition
P = sp.csc_matrix([[2, 0], [0, 2]])
q = np.array([0, 0])

A = sp.csc_matrix([-2, 1])
l = np.array([-np.Inf])
u = np.array([-1])

# Solution
model = osqp.OSQP()
model.setup(
    P=P, q=q,
    A=A, l=l, u=u,
    verbose=False
)
results = model.solve()

# Visualization and printout
print("x =", results.x)
print("f(x) =", results.info.obj_val)

plt.scatter(results.x[0], results.x[1], color="green")
plt.show()
