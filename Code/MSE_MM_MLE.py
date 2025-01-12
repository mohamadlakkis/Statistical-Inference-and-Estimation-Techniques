
import numpy as np 
from scipy.special import gamma
import matplotlib.pyplot as plt 
from scipy.special import gammaln
N = np.arange(1, 150 , 1)
def A(n):
    
    return gamma(n+0.5)/gamma(n)
THETA= 2
MSE = []
MM = []
for n in N: 
    MSE.append(THETA**2*(2-2*A(n)/np.sqrt(n)))
    MM.append(THETA**2*((4-np.pi)/(2)))
plt.figure(figsize=(8, 6))
plt.plot(N, MSE, label = "MLE", linestyle = "-",color='blue')
plt.plot(N, MM, label = "MOM", linestyle = "--",color='red')
plt.title("Comparing MLE and MOM based on MSE")
plt.xlabel("n")
plt.yscale('log')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
    