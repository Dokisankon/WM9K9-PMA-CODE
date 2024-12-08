import numpy as np
import pandas as pd
import math
from scipy.stats import *
import seaborn as sns
import matplotlib.pyplot as plt

## (i)(a)
mean = 2.6
x = 2

prob = (mean**x * np.exp(-mean))/(math.factorial(x))

print(f"The probability of having {x} defect in 1 meter is {prob:.4f}")

## (i)(b)
def poisson_prob(mean, x):
    return(mean**x * np.exp(-mean))/(math.factorial(x))

prob = 1
for x in range(9):
    prob -= poisson_prob(mean = 2.6*5, x = x)
    
print(f"The probability of having more than 8 defect in 5 meter is {prob:.4f}")

##(i)(c)
prob = 0
for x in range(20):
    prob += poisson_prob(mean = 2.6*10, x = x)
    
print(f"The probability of having less than 20 defect in 10 meter is {prob:.4f}")


##(ii)(a)
data = np.array([33.75, 33.05, 34.00, 33.81, 33.46, 34.02, 33.58, 33.27, 33.49, 33.20, 34.62, 33.00, 33.54, 34.12, 33.84])

print(data.mean())

##(ii)(b)

diff_norm = norm.ppf(0.975)
std = np.power(data.var(), 0.5)

diff_range = diff_norm * (std / np.sqrt(15))
print(f"The upper bound of data is: {(data.mean() + diff_range):.4f}")
print(f"The lower bound of data is: {(data.mean() - diff_range):.4f}")

##(ii)(c)
mean = np.mean(data)
std = np.std(data)

num_bins = int(1 + 3.322 * np.log10(len(data)))

## Show the historgram
fig = plt.figure()
ax = fig.add_subplot(111)
ax = sns.histplot(data, bins = num_bins)
plt.title('Historgram of viscosity')
# plt.show()

hist, bin_edges = np.histogram(data, bins = num_bins)

modify_bins = bin_edges
modify_bins[0] = 0
modify_bins[-1]= 99999
expected_probs = norm.cdf(modify_bins, mean, std)
expected_probs = expected_probs[1:] - expected_probs[:-1]

expected = np.array(expected_probs) * len(data)

chi2_stat, p_value = chisquare(f_obs=hist, f_exp=expected)
print(f"Chi-value: {chi2_stat}")
print(f"p-value: {p_value}")

temp =  norm.cdf([33,35], mean, std)
print(temp)
print(f"The probability if smaller than 33 is: {temp[0]:.4f}")
print(f"The probability if larger than 35 is: {1-temp[1]:.4f}")
print(f"The probability if in outside is: {(temp[0] + 1-temp[1]):.4f}")
print()

plt.show()