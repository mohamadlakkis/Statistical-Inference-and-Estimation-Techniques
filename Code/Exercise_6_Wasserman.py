import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt

mu = 5
n = 100
X = np.random.normal(mu, 1, n)


theta = np.exp(mu)
theta_hat = np.exp(X.mean()) 

B = 10000
theta_hat_star = np.zeros(B)
for b in range(B):
    X_star = np.random.choice(X, n, replace=True)
    theta_hat_star[b] = np.exp(X_star.mean())
se_theta_hat_boot = np.std(theta_hat_star)

# we will use Normal CI 
alpha = 0.05
z_alpha = norm.ppf(1 - alpha/2)

Normal_CI = (theta_hat - z_alpha*se_theta_hat_boot, theta_hat + z_alpha*se_theta_hat_boot)

print("######################################################PART A ###########################################################")
print("True Theta: ", theta)
print("Estimated Normal CI: ", Normal_CI)


'''Part B'''
# True sampling distribution of theta_hat
N = 10000
theta_hat_True = np.zeros(N)
for i in range(N):
    X = np.random.normal(mu, 1, n)
    theta_hat_True[i] = np.exp(X.mean())



plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.hist(theta_hat_star, bins=50, color = 'blue',edgecolor='black',density=True,alpha = 0.7)
plt.title(r'Bootstrap Distribution of $\hat{\theta}$')

plt.subplot(1, 2, 2)
plt.hist(theta_hat_True, bins=50, color = 'red',edgecolor='black',density=True,alpha = 0.7)
plt.title(r'True Sampling Distribution of $\hat{\theta}$')
plt.tight_layout()
plt.show()