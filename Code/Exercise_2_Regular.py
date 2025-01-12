import numpy as np
from scipy.stats import skew, norm
'''Skewness from Monte Carlo Simulation'''
# n = 100000 # number of samples per set
# N = 1000  # number of sets
# Y = np.random.normal(0, 1, (N, n))
# X = np.exp(Y) 
# skewness_values = skew(X, axis=1) 
# print("Skewness from Monte Carlo Simulation is: ",skewness_values.mean()) 


Theta = 6.18 # True Skewness

def calc_skew_sample(X):
    n, mu, sigma = X.size, X.mean(), X.std()
    tmp = 0
    for x_i in X: 
        tmp += (x_i - mu)**3
    return tmp/(n * sigma**3)


Number_of_simulations = 100
n = 50
B = 1000
count_normal = 0
count_pivotal = 0
count_percentile = 0

Normal_first_bound = []
Normal_second_bound = []
Pivotal_first_bound = []
Pivotal_second_bound = []
Percentile_first_bound = []
Percentile_second_bound = []

for i in range(Number_of_simulations):
    Y = np.random.normal(0, 1, n)
    X = np.exp(Y)
    theta_hat = calc_skew_sample(X)
    # Let's do the bootstrap: to estimate  the se(theta_hat) and to get the Bootstrap samples(which we will use in the methods for computing CI )
    theta_hat_star = np.zeros(B) # to store the bootstrap estimates
    for b in range(B):
        X_star = np.random.choice(X, n, replace=True) # mass at each point is 1/n, ->  Empirical distribution
        theta_hat_star[b] = calc_skew_sample(X_star)
    se_hat_theta_boot = np.std(theta_hat_star)

    '''now we will compute the Normal Confidence Interval with alpha = 0.05 -> 95% Confidence Interval'''
    alpha = 0.05 
    z_alpha =  norm.ppf(1 - alpha/2) # or abs(norm.ppf(alpha/2)) (symmetric)
    Normal_CI = (theta_hat - z_alpha*se_hat_theta_boot, theta_hat + z_alpha*se_hat_theta_boot)
    if Normal_CI[0] <= Theta <= Normal_CI[1]:
        count_normal += 1
    Normal_first_bound.append(Normal_CI[0])
    Normal_second_bound.append(Normal_CI[1])
    '''Bootrap pivotal CI'''
    Pivotal_CI = (2*theta_hat-np.quantile(theta_hat_star,1-alpha/2), 2*theta_hat-np.quantile(theta_hat_star,alpha/2))
    if Pivotal_CI[0] <= Theta <= Pivotal_CI[1]:
        count_pivotal += 1
    Pivotal_first_bound.append(Pivotal_CI[0])
    Pivotal_second_bound.append(Pivotal_CI[1])
    '''Bootsrap Percentile CI'''
    Percentile_CI = (np.quantile(theta_hat_star,alpha/2), np.quantile(theta_hat_star,1-alpha/2))
    if Percentile_CI[0] <= Theta <= Percentile_CI[1]:
        count_percentile += 1
    Percentile_first_bound.append(Percentile_CI[0])
    Percentile_second_bound.append(Percentile_CI[1])
print("The coverage probability for Normal Confidence Interval is: ", count_normal/Number_of_simulations) # estimation of P(Theta in CI_Normal)
print("The coverage probability for Pivotal Confidence Interval is: ", count_pivotal/Number_of_simulations) # estimation of P(Theta in CI_Pivotal)
print("The coverage probability for Percentile Confidence Interval is: ", count_percentile/Number_of_simulations) # estimation of P(Theta in CI_Percentile)

print("Now the estimation of Normal CI is: ", (np.mean(Normal_first_bound), np.mean(Normal_second_bound)))
print("Now the estimation of Pivotal CI is: ", (np.mean(Pivotal_first_bound), np.mean(Pivotal_second_bound)))
print("Now the estimation of Percentile CI is: ", (np.mean(Percentile_first_bound), np.mean(Percentile_second_bound)))