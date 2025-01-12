import numpy as np
import matplotlib.pyplot as plt

theta = 1
n_values = [100, 1000, 10000, 100000]
N = 1000
epsilon = 0.05  
prob = []
for n in n_values:
    theta_hat_MLE_samples = []
    for i in range(N):
        X_squared = np.random.exponential(scale=2 * theta**2, size=n)
        theta_hat_MLE = np.sqrt((1/2) * np.mean(X_squared))
        theta_hat_MLE_samples.append(theta_hat_MLE)
    theta_hat_MLE_samples = np.array(theta_hat_MLE_samples)
    pp = np.mean(theta_hat_MLE_samples - theta > epsilon)
    prob.append(pp)

plt.figure(figsize=(16, 8))
plt.plot(n_values, prob, marker='o')
plt.xscale('log')
plt.title("P(|Theta_MLE - Theta|>epsilon) VS n")
plt.xlabel("n")
plt.ylabel("Probability")
plt.show()
