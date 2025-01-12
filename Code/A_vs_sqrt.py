import numpy as np 
from scipy.special import gamma
import matplotlib.pyplot as plt 
from scipy.special import gammaln
N = np.arange(1, 40 , 1)
def A(n):
    return gamma(n+0.5)/gamma(n)
AA = []
sqrt_n = []
for n in N: 
    AA.append(A(n))
    sqrt_n.append(np.sqrt(n))
plt.figure(figsize=(8, 6))
plt.plot(N, AA, label = "A", linestyle = "-",color='blue')
plt.plot(N, sqrt_n, label = "sqrt(n)", linestyle = "--",color='red')
plt.title("Comparing A and sqrt(n)")
plt.xlabel("n")
plt.yscale('log')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
    