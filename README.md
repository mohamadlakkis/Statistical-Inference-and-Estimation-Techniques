# Statistical Inference and Estimation Techniques

## Overview
This repository contains code and analysis related to statistical inference and estimation techniques, focusing on confidence intervals, bootstrap methods, and additional topics such as UMVUE, exponential families, and MLE analysis. The main topics covered include:

1. **Bootstrap Confidence Intervals**: Estimating the coverage of three types of confidence intervals for the true skewness of a population.
2. **Exact and Monte Carlo Approximation**: Calculating the true skewness using theoretical and Monte Carlo methods.
3. **Confidence Intervals**:
  - **Normal Confidence Interval**: Calculated as \(\hat{\theta} \pm z_{\alpha/2} \hat{\sigma}_{boot}\).
  - **Pivotal Confidence Interval**: Calculated as \(C_n = \left( 2\hat{\theta} - \hat{\theta}^*_{1-\alpha/2}, \, 2\hat{\theta} - \hat{\theta}^*_{\alpha/2} \right)\).
  - **Percentile Confidence Interval**: Calculated as \(C_n = \left( \hat{\theta}^*_{\alpha/2}, \, \hat{\theta}^*_{1-\alpha/2} \right)\).
4. **UMVUE, Exponential Families, and MLE Analysis**: Conducted in-depth studies on sufficiency, exponential families, and the Rao-Blackwell theorem, including construction and analysis of UMVUEs, deriving sufficient statistics, and proving the efficiency of estimators under the Cramér-Rao bound. Analyzed MLEs for exponential families, proving optimality and constructing transformations to achieve minimal variance estimators. Compared parametric and non-parametric estimators through simulations across different distributions, evaluating their performance in terms of MSE.

## Key Findings
- With a small sample size (n=50), none of the confidence intervals provided good estimates.
- Increasing the sample size improves the estimates, as larger samples provide better approximations of the true distribution.

## Code and Analysis
- **Exercise_2_Regular.py**: Code for calculating the skewness and confidence intervals.
- **Exercise_2_many_n.py**: Code for analyzing the effect of increasing sample size on confidence intervals.
- **Exercise_6_Wasserman.py**: Code for additional exercises from Wasserman's book.

## Results
- The analysis shows that larger sample sizes yield better estimates of the true skewness.
- The bootstrap method provides a good approximation of the distribution of the estimator.

## Conclusion
This repository demonstrates the application of bootstrap methods and confidence interval estimation in statistical inference. The results highlight the importance of sample size in obtaining accurate estimates.


For more details, refer to the code and analysis provided in the repository. If you have any questions or feedback, please feel free to reach out at mohamad.allakkis@gmail.com
