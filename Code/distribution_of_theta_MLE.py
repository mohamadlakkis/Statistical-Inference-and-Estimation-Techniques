import numpy as np 
import matplotlib.pyplot as plt

theta = 1

n = 100000
NN = 10000
distribution_theta_MLE = []
for N in range(NN):
    X_Sqaured = np.random.exponential(scale=(2*theta**2),size= n)# note scale = 1/lambda
    theta_hat_MLE = np.sqrt((1/2)* np.mean(X_Sqaured))
    distribution_theta_MLE.append(np.sqrt(n)*(theta_hat_MLE-theta))
Normal_dist = np.random.normal(loc = 0, scale=theta /2, size=NN) # note scale = sigma 

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.hist(distribution_theta_MLE, bins=30, color='blue', alpha=0.7, label='Distribution of Theta_MLE',density=True)
plt.title("Distribution of Sqrt(n)[Theta_MLE-THETA]")
plt.xlabel("Observations")
plt.ylabel("Density")
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(Normal_dist, bins=30, color='red', alpha=0.7, label='Normal Distribution, N(0,theta^2/4)',density=True)
plt.title("Normal Distribution")
plt.xlabel("Observations")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()