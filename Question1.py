import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import *
import seaborn as sns

## Load data
data = {
    "County"    : ['Clatsop', 'Columbia', 'Gilliam', 'Hood River', 'Morrow', 'Portland', 'Sherman', 'Umatilla', 'Wasco'],
    "REI"       : [8.3, 6.4,3.4,3.8,2.6,11.6,1.2,2.5,1.6],
    "Mortality" : [210,180,130,170,130,210,120,150,140]
}

df = pd.DataFrame(data)
df = df.sort_values('REI').reset_index(drop = True)

print(df)

### (a)
## getting the figure 1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data['REI'], data['Mortality'], marker = 'o')
plt.xlabel('REI')
plt.ylabel('Mortality')
# plt.show()
# exit()

pearson_corr, p_value = pearsonr(df['REI'], df['Mortality'])

print()
print(f'The pearson correlation value is: {pearson_corr}')
print(f'The P-value of this test is: {p_value}')
print()
'''
Null hypothesis(H0):The correlation coefficient were in fact zero
Alternative hypothesis(H1):The correlation coefficient were not zero
'''

if p_value > 0.05:
    print("They are not Related.")
else:
    print("They are Related.")

## (b)
## Do the linear regression.
import statsmodels.api as sm

X = sm.add_constant(data['REI'])
y = data['Mortality']

print(X)
print(y)
# exit()
model = sm.OLS(y, X)
model = model.fit()
intercept, slope = model.params

print(model.summary())

print(f"The predicted value of 5 REI is {model.predict([1,5])}")
print(f"The intercept is {intercept}")
print(f"The slope is {slope}")

ypoints = np.array(range(int(round(max(df['REI']), 0))+1)) * slope + intercept
plt.plot(ypoints)
# plt.title('Predict line of Mortality')
plt.xlabel('REI')
plt.ylabel('Mortality')
plt.show()

