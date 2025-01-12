import numpy as np
import pandas as pd
import itertools
from scipy.stats import expon
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

taus = [0, 1, 2, 4]   
lambdas = [0.1, 0.5, 1, 2, 4] 
ns = [5, 10, 50, 100]        
alpha = 0.01  
M = 10000   # used later while getting the coverage probabilities

results = []
parameter_combinations = list(itertools.product(taus, lambdas, ns)) # all combo 

for tau, lam, n in parameter_combinations:
    
    coverage_CI1 = 0  # CI 1 is using lambda UMVUE estimator (estimation of the true coverage probability)
    coverage_CI2 = 0  # CI 2 is using true lambda (estimation of the true coverage probability)
    lengths_CI1 = [] # for each iteration m, we will store the length of CI1_m, and the take the mean, which we will compare with the average length of CI2
    lengths_CI2 = [] # Smae thing here bbut for CI2_m
    
    
    for _ in range(M):
        '''Generate n samples from the shifted exponential distribution'''
        Y = np.random.exponential(scale=1/lam, size=n)  # Exp(lam)
        W = tau + Y 
        
        W_min = np.min(W)
        
        '''Calculate  lambda_UMVUE estimator'''
        sum_differences = np.sum(W - W_min)
        
        lambda_hat = (n - 2) / sum_differences
        
        '''Calculate the confidence intervals (CI1)'''

        lower_bound_CI1 = W_min + (np.log(alpha)) / (n * lambda_hat)
        upper_bound_CI1 = W_min 
        
        #Check if the true tau is within CI1, this step is done to approximate the coverage probability
        if lower_bound_CI1 <= tau <= upper_bound_CI1:
            coverage_CI1 += 1 
        # Calculate the length of CI1
        length_CI1 = upper_bound_CI1 - lower_bound_CI1
        lengths_CI1.append(length_CI1) 
        
        '''Second confidence interval (CI2) using true lambda:'''

        lower_bound_CI2 = W_min + (np.log(alpha)) / (n * lam)
        upper_bound_CI2 = W_min  
        
        #Check if the true tau is within CI2, this step is done to approximate the coverage probability of CI2
        if lower_bound_CI2 <= tau <= upper_bound_CI2:
            coverage_CI2 += 1  
        # Calculate the length of CI2
        length_CI2 = upper_bound_CI2 - lower_bound_CI2
        lengths_CI2.append(length_CI2)
    

    '''Calculate the estimated coverage probabilities and average lengths'''
    coverage_prob_CI1 = coverage_CI1 / M
    average_length_CI1 = np.mean(lengths_CI1)
    
    coverage_prob_CI2 = coverage_CI2 / M
    average_length_CI2 = np.mean(lengths_CI2)
    
    '''Store the results'''
    results.append({
        'tau': tau,
        'lambda': lam,
        'n': n,
        'coverage_CI1': coverage_prob_CI1 * 100,  
        'average_length_CI1': average_length_CI1,
        'coverage_CI2': coverage_prob_CI2 * 100, 
        'average_length_CI2': average_length_CI2,
        'nominal_coverage': (1 - alpha) * 100   
    })
    

# for visualization purposes 
results_df = pd.DataFrame(results)
print(results_df)


'''##################################'''
'''1. for coverage probabilities'''
plt.figure(figsize=(14, 7))
coverage_df = results_df.melt(id_vars=['tau', 'lambda', 'n', 'nominal_coverage'],
                              value_vars=['coverage_CI1', 'coverage_CI2'],
                              var_name='CI', value_name='Coverage')
sns.barplot(data=coverage_df, x='n', y='Coverage', hue='CI')
plt.axhline(y=(1 - alpha) * 100, color='red', linestyle='--', label='Nominal Coverage')
plt.title('Coverage Probability Comparison')
plt.xlabel('Sample Size (n)')
plt.ylabel('Coverage Probability (%)')
plt.legend(title='Confidence Interval')
plt.savefig('CI_Coverage_prob_1_2.png')
plt.show()


'''##################################'''
'''2. Pick 4 combinations of tau and lambda to compare lengths over n for both CI1 and CI2 on the same graph'''
selected_combinations = [(0, 4), (1, 4), (2, 4), (4, 4)]
fig, ax = plt.subplots(figsize=(10, 6))

for tau_lam in selected_combinations:
    tau_val, lam_val = tau_lam
    subset_CI1 = results_df[(results_df['tau'] == tau_val) & (results_df['lambda'] == lam_val)]
    subset_CI2 = results_df[(results_df['tau'] == tau_val) & (results_df['lambda'] == lam_val)]
    ax.plot(subset_CI1['n'], subset_CI1['average_length_CI1'], marker='o', label=f'CI1 tau={tau_val}, lam={lam_val}')
    ax.plot(subset_CI2['n'], subset_CI2['average_length_CI2'], marker='x', linestyle='--', label=f'CI2 tau={tau_val}, lam={lam_val}')

ax.set_title(r'Comparison of CI Lengths over n for Selected \tau and \lambda')
ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('Average Length of CI')
ax.legend()
plt.savefig('CI_Lengths_Comparison.png')
plt.show()

'''##################################'''
'''3. Additional comparison: Plot the difference in lengths between CI1 and CI2 over n'''
fig, ax = plt.subplots(figsize=(10, 6))

for tau_lam in selected_combinations:
    tau_val, lam_val = tau_lam
    subset = results_df[(results_df['tau'] == tau_val) & (results_df['lambda'] == lam_val)]
    length_diff = subset['average_length_CI1'] - subset['average_length_CI2']
    ax.plot(subset['n'], length_diff, marker='o', label=f'Delta_Length tau={tau_val}, lamb={lam_val}')

ax.set_title('Difference in Average Lengths (CI1 - CI2) over n')
ax.set_xlabel('Sample Size (n)')
ax.set_ylabel('Difference in Lengths')
ax.legend()
plt.savefig('CI_Lengths_Difference.png')
plt.show()
