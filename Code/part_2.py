import numpy as np
import scipy.stats as stats
np.random.seed(42)

'''Function to estimate lambda using UMVUE'''
def estimate_lambda_umvue(W):
    n = len(W)
    W_min = np.min(W)
    sum_differences = np.sum(W - W_min)
    lambda_hat = (n - 2) / sum_differences
    return lambda_hat

''' Function to compute CI1'''
def compute_CI1(W, alpha):
    n = len(W)
    W_min = np.min(W)
    lambda_hat = estimate_lambda_umvue(W)
    if np.isnan(lambda_hat) or lambda_hat <= 0:
        return (np.nan, np.nan)  # Return NaNs if estimate is invalid
    lower_bound = W_min + np.log(alpha) / (n * lambda_hat)
    upper_bound = W_min
    return (lower_bound, upper_bound)

# Bootstrap function ( very similar to the one in Exercise 6)
def bootstrap_CIs(W, alpha, B=1000):
    '''Here we are taking W_min as an estimator of tau'''
    n = len(W)
    W_min = np.min(W)
    tau_hat = W_min  # Estimator of tau

    tau_bootstrap = []
    lambda_bootstrap = []

    # B bootstrap samples
    for _ in range(B):
        '''with replacement'''
        W_bootstrap = np.random.choice(W, size=n, replace=True)
        W_min_bootstrap = np.min(W_bootstrap)
        tau_hat_bootstrap = W_min_bootstrap
        tau_bootstrap.append(tau_hat_bootstrap)

    # numpy arrays
    tau_bootstrap = np.array(tau_bootstrap)
    lambda_bootstrap = np.array(lambda_bootstrap)


    '''Normal interval'''
    SE_bootstrap = np.std(tau_bootstrap, ddof=1)
    z = stats.norm.ppf(1 - alpha / 2)
    CI_normal = (tau_hat - z * SE_bootstrap, tau_hat + z * SE_bootstrap)

    '''Percentile interval'''
    lower_percentile = np.percentile(tau_bootstrap, 100 * alpha / 2)
    upper_percentile = np.percentile(tau_bootstrap, 100 * (1 - alpha / 2))
    CI_percentile = (lower_percentile, upper_percentile)

    '''Pivotal interval'''
    lower = 2* tau_hat - upper_percentile
    upper = 2* tau_hat - lower_percentile
    CI_pivotal = (lower, upper)
    return {
        'CI_normal': CI_normal,
        'CI_percentile': CI_percentile,
        'CI_pivotal': CI_pivotal
    }
# Function to read numbers from a file and return them as a numpy array
def read_numbers_from_file(filename):
    with open(filename, 'r') as file:
        numbers = [float(line.strip()) for line in file]
    return np.array(numbers)


alpha = 0.05     
W = read_numbers_from_file('Exam_2023_last_problem_part_10/physicslab.txt')
'''CI 1'''
print(f"\nthe W_min values = {np.min(W)}",end = "\n\n") 

CI1 = compute_CI1(W, alpha)
print(f"\nCI1 (Using UMVUE estimate): {CI1}",end = "\n\n")

'''Bootstrap confidence intervals'''

bootstrap_results = bootstrap_CIs(W, alpha)
print(f"Bootstrap Pivotal Interval: {bootstrap_results['CI_pivotal']}",end="\n\n")
print(f"Bootstrap Normal Interval: {bootstrap_results['CI_normal']}",end="\n\n")
print(f"Bootstrap Percentile Interval: {bootstrap_results['CI_percentile']}")


