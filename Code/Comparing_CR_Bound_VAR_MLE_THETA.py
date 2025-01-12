
import numpy as np 
from scipy.special import gamma
import matplotlib.pyplot as plt 
from scipy.special import gammaln
N = np.arange(1, 10 , 1)
def A(n):
    '''Note I am using gammaln just for computations, it is the same once we re apply the exponential'''
    return np.exp(gammaln(n + 1/2) - gammaln(n))
LHS = []
RHS = []
for n in N: 
    a_n = A(n)
    lhs = 1- a_n**2/n
    LHS.append(lhs)
    rhs = a_n**2/(4*n**2)
    RHS.append(rhs)
plt.figure(figsize=(8, 6))
plt.plot(N, LHS, label = "Variance of MLE", linestyle = "-")
plt.plot(N, RHS, label = "CR Bound", linestyle = "--")
plt.title("Comparing CR Bound and Variance of MLE")
plt.xlabel("n")
plt.yticks([])
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
    