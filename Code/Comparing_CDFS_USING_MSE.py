import numpy as np 
import scipy 
import matplotlib.pyplot as plt

def F_UMVUE(u,n,X_BAR_n):
    VAL = np.sqrt(n/(n-1)) * (u-X_BAR_n)
    return scipy.stats.norm.cdf(VAL)

def F_MLE(u,X_BAR_n):
    return scipy.stats.norm.cdf(u-X_BAR_n)

def F_NP(u,X_vector):
    return np.mean(X_vector < u)

def F_X_norm(u,mu,sigma):
    return scipy.stats.norm.cdf(u,mu,sigma)

def F_X_exp(u):
    return scipy.stats.expon.cdf(u) # by default scale = 1, (lambda = 1/scale = 1, as we want!)
SETUP = {
    "X_i ~ N(0,1)": ("NORMAL", 0,1, [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]),
    "X_i ~ N(0,sigma^2=4)" : ("NORMAL", 0,2,[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]),
    "X_i ~ EXP(1)" : ("EXP", [0.1, 0.5, 1, 1.5, 2, 2.5]),
}

fig, axes = plt.subplots(len(SETUP), 1, figsize=(12, 18))
fig.subplots_adjust(hspace=0.4)
N = [5,10,50,200]
for i,set in enumerate(SETUP):
    MSE_UMVUE = []
    MSE_MLE = []
    MSE_NP = []
    if SETUP[set][0] == "NORMAL":
        mu = SETUP[set][1]
        sigma = SETUP[set][2]
        u = SETUP[set][3]
        for n in N:
            # getting our fixed sample of size n
            X_vector = np.random.normal(mu,sigma,n)
            X_BAR_n = np.mean(X_vector)
            mse_UMVUE = 0
            mse_MLE = 0
            mse_NP = 0
            for u_val in u:
                TRUE_F = F_X_norm(u_val,mu,sigma)
                mse_UMVUE += (TRUE_F - F_UMVUE(u_val,n,X_BAR_n))**2
                mse_MLE += (TRUE_F - F_MLE(u_val,X_BAR_n))**2
                mse_NP += (TRUE_F - F_NP(u_val,X_vector))**2
            mse_MLE /= len(u)
            mse_UMVUE /= len(u)
            mse_NP /= len(u)
            MSE_UMVUE.append(mse_UMVUE)
            MSE_MLE.append(mse_MLE)
            MSE_NP.append(mse_NP)
    else:
        u = SETUP[set][1]
        for n in N:
            # getting our fixed sample of size n
            X_vector = np.random.exponential(1,n)
            X_BAR_n = np.mean(X_vector)
            mse_UMVUE = 0
            mse_MLE = 0
            mse_NP = 0
            for u_val in u:
                TRUE_F = F_X_exp(u_val)
                mse_UMVUE += (TRUE_F - F_UMVUE(u_val,n,X_BAR_n))**2
                mse_MLE += (TRUE_F - F_MLE(u_val,X_BAR_n))**2
                mse_NP += (TRUE_F - F_NP(u_val,X_vector))**2
            mse_MLE /= len(u)
            mse_UMVUE /= len(u)
            mse_NP /= len(u)
            MSE_UMVUE.append(mse_UMVUE)
            MSE_MLE.append(mse_MLE)
            MSE_NP.append(mse_NP)
        
    ax = axes[i]
    ax.plot(N, MSE_UMVUE, label='UMVUE', marker='o')
    ax.plot(N, MSE_MLE, label='MLE', marker='s')
    ax.plot(N, MSE_NP, label='NP', marker='^')
    ax.set_title(f'MSE Comparison for {set}', fontsize=8)
    if i==len(SETUP)-1:
        ax.set_xlabel('Sample Size (n)')
        
    else:
        ax.set_xticks([])
    if i==0:
        ax.legend()
    ax.set_ylabel('MSE')
plt.tight_layout()
plt.show()