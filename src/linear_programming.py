import cvxpy as cp
import numpy as np

A = np.array([[-1, 1], [1, 6], [4, -1]])
c = np.array([1, 1])
b = np.array([1, 15, 10])

# Primal problem
x = cp.Variable(2)

primal_objective = cp.Maximize(c.T @ x)
primal_constraints = [x >= 0, A @ x <= b]

primal = cp.Problem(primal_objective, primal_constraints)
primal.solve(solver=cp.GLPK)

print("x = ", x.value)
print("c'x = ", primal.value)

# Dual problem
y = cp.Variable(3)

dual_objective = cp.Minimize(b.T @ y)
dual_constraints = [y >= 0, A.T @ y >= c]

dual = cp.Problem(dual_objective, dual_constraints)
dual.solve(solver=cp.GLPK)

print("y = ", y.value)
print("b'y = ", dual.value)
