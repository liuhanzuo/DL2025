import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

import matplotlib.pyplot as plt

# Define the target distribution p(x)
def target_distribution(x):
    return (1/3) * norm.pdf(x, loc=-3, scale=1) + (2/3) * norm.pdf(x, loc=3, scale=1)

# Inclusive KL divergence (KL(q || p))
def inclusive_kl(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q = norm(loc=mu, scale=sigma)
    x = np.linspace(-10, 10, 1000)
    kl = np.sum(q.pdf(x) * (np.log(q.pdf(x)) - np.log(target_distribution(x)))) * (x[1] - x[0])
    return kl

# Exclusive KL divergence (KL(p || q))
def exclusive_kl(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    q = norm(loc=mu, scale=sigma)
    x = np.linspace(-10, 10, 1000)
    kl = np.sum(target_distribution(x) * (np.log(target_distribution(x)) - np.log(q.pdf(x)))) * (x[1] - x[0])
    return kl

# Optimize for inclusive KL
inclusive_result = minimize(inclusive_kl, [0, 0], method='L-BFGS-B')
mu_inclusive, log_sigma_inclusive = inclusive_result.x
sigma_inclusive = np.exp(log_sigma_inclusive)

# Optimize for exclusive KL
exclusive_result = minimize(exclusive_kl, [0, 0], method='L-BFGS-B')
mu_exclusive, log_sigma_exclusive = exclusive_result.x
sigma_exclusive = np.exp(log_sigma_exclusive)

# Plot the results
x = np.linspace(-10, 10, 1000)
p_x = target_distribution(x)
q_inclusive = norm.pdf(x, loc=mu_inclusive, scale=sigma_inclusive)
q_exclusive = norm.pdf(x, loc=mu_exclusive, scale=sigma_exclusive)

plt.figure(figsize=(10, 6))
plt.plot(x, p_x, label="Target Distribution p(x)", color="black", linewidth=2)
plt.plot(x, q_inclusive, label=f"Variational q(x) (Inclusive KL)\nμ={mu_inclusive:.2f}, σ={sigma_inclusive:.2f}", color="blue")
plt.plot(x, q_exclusive, label=f"Variational q(x) (Exclusive KL)\nμ={mu_exclusive:.2f}, σ={sigma_exclusive:.2f}", color="red")
plt.title("Target and Variational Distributions")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.savefig("6-3.png")
plt.show()