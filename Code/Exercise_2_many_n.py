import numpy as np
from scipy.stats import skew, norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def calc_skew_sample(X):
    n, mu, sigma = X.size, X.mean(), X.std()
    tmp = 0
    for x_i in X:
        tmp += (x_i - mu)**3
    return tmp / (n * sigma**3)


def run_simulation(seed, Theta=6.18, n=1000, B=100):
    np.random.seed(seed)  
    Y = np.random.normal(0, 1, n)
    X = np.exp(Y)
    theta_hat = skew(X)

    theta_hat_star = np.zeros(B)
    for b in range(B):
        X_star = np.random.choice(X, n, replace=True)
        theta_hat_star[b] = skew(X_star)
    se_hat_theta_boot = np.std(theta_hat_star)

    alpha = 0.05
    z_alpha = norm.ppf(1 - alpha / 2)
    Normal_CI = (theta_hat - z_alpha * se_hat_theta_boot, theta_hat + z_alpha * se_hat_theta_boot)

    Pivotal_CI = (2 * theta_hat - np.quantile(theta_hat_star, 1 - alpha / 2),
                  2 * theta_hat - np.quantile(theta_hat_star, alpha / 2))

    Percentile_CI = (np.quantile(theta_hat_star, alpha / 2), np.quantile(theta_hat_star, 1 - alpha / 2))

    result = {
        "normal_ci": Normal_CI,
        "pivotal_ci": Pivotal_CI,
        "percentile_ci": Percentile_CI,
        "normal_coverage": Normal_CI[0] <= Theta <= Normal_CI[1],
        "pivotal_coverage": Pivotal_CI[0] <= Theta <= Pivotal_CI[1],
        "percentile_coverage": Percentile_CI[0] <= Theta <= Percentile_CI[1]
    }

    return result


def main():
    Number_of_simulations = 1000
    B = 1000000
    Theta = 6.18
    n_values = [50, 100, 500, 1000, 3000,5000,10000] 


    normal_probabilities = []
    pivotal_probabilities = []
    percentile_probabilities = []

    with open('bootstrap_bounds.txt', 'w') as f:

        for n in n_values:
            seeds = np.random.randint(0, 10000, Number_of_simulations)

            results = Parallel(n_jobs=-1)(delayed(run_simulation)(seed, Theta, n, B) for seed in seeds)

            count_normal = sum([res["normal_coverage"] for res in results])
            count_pivotal = sum([res["pivotal_coverage"] for res in results])
            count_percentile = sum([res["percentile_coverage"] for res in results])

            normal_first_bound = [res["normal_ci"][0] for res in results]
            normal_second_bound = [res["normal_ci"][1] for res in results]
            pivotal_first_bound = [res["pivotal_ci"][0] for res in results]
            pivotal_second_bound = [res["pivotal_ci"][1] for res in results]
            percentile_first_bound = [res["percentile_ci"][0] for res in results]
            percentile_second_bound = [res["percentile_ci"][1] for res in results]

            f.write(f"n = {n}\n")
            f.write(f"Normal CI bounds: {normal_first_bound} \n")  
            f.write(f"Normal CI bounds: {normal_second_bound} \n")  
            f.write(f"Pivotal CI bounds: {pivotal_first_bound} \n")  
            f.write(f"Pivotal CI bounds: {pivotal_second_bound} \n")  
            f.write(f"Percentile CI bounds: {percentile_first_bound} \n") 
            f.write(f"Percentile CI bounds: {percentile_second_bound} \n")  

            normal_probabilities.append(count_normal / Number_of_simulations)
            pivotal_probabilities.append(count_pivotal / Number_of_simulations)
            percentile_probabilities.append(count_percentile / Number_of_simulations)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, normal_probabilities, label='Normal CI', marker='o')
    plt.plot(n_values, pivotal_probabilities, label='Pivotal CI', marker='o')
    plt.plot(n_values, percentile_probabilities, label='Percentile CI', marker='o')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Coverage Probability')
    plt.title('Coverage Probability vs Sample Size for Different CIs')
    plt.legend()
    plt.grid(True)
    plt.savefig('coverage_probabilities_plot.png')
    plt.show()


main()
