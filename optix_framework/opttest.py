import scipy.optimize
import numpy as np

def f(x):   # The rosenbrock function
	return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
	

xopt, fopt , iter , funcalls , warnflag, allvecs = scipy.optimize.fmin(f, [2, 2], full_output=1, retall=True)

data = np.array(allvecs)

import matplotlib.pyplot as plt
plt.plot(data[:,0], label='x0')
plt.plot(data[:,1], label='x1')
plt.ylabel('Value')
plt.legend()
plt.show()