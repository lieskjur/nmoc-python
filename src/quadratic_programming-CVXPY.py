import cvxpy as cp
import numpy as np
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
Q = np.array([[2, 0], [0, 2]])
c = np.array([0, 0])

A = np.array([-2, 1])
b = np.array([-1])

# Solution
x = cp.Variable(2)

objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)
constraints = [A @ x <= b]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP)

# Visualization and printout
print("x = ", x.value)
print("f(x) = ", problem.value)

plt.scatter(x.value[0], x.value[1], color="green")
plt.show()
