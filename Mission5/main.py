import numpy as np
import matplotlib.pyplot as plt
from BenchmarkFunctions import BenchmarkFunctions
from RUN import RUN

# Assuming the RUN function and BenchmarkFunctions have been defined as previously provided.

# Setup the parameters
nP = 50                # Number of Population
Func_name = 'F1'       # Name of the test function, range from F1-F14
MaxIt = 500            # Maximum number of iterations

# Load details of the selected benchmark function
lb, ub, dim, fobj = BenchmarkFunctions(Func_name)

# Execute the optimization algorithm
Best_fitness, BestPositions, Convergence_curve = RUN(nP, MaxIt, lb, ub, dim, fobj)

# Draw the objective space
plt.figure()
plt.semilogy(Convergence_curve, color='r', linewidth=4)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.tight_layout()
plt.grid(True)
plt.legend(['RUN'])
plt.show()
