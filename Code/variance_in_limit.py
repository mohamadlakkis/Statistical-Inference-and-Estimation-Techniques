import numpy as np
lambda_true = 5  
theta_true = 1 / lambda_true  
n_simulations = 10000
theta_mle_simulations = []
n = 100000
for _ in range(n_simulations):
    X_samples = np.random.poisson(lambda_true, n)
    lambda_mle = np.mean(X_samples)
    theta_mle = 1 / lambda_mle
    theta_mle_simulations.append(theta_mle)
VAR = np.var(theta_mle_simulations)
print("n * var(theta_MLE) = ",n * VAR, " theta^3 = ",theta_true**3)
print("var(theta_MLE) = ",VAR, " theta^3 = ",theta_true**3)