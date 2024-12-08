import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import *
import seaborn as sns
# import plotly.express as px

data = {
    "County"    : ['Clatsop', 'Columbia', 'Gilliam', 'Hood River', 'Morrow', 'Portland', 'Sherman', 'Umatilla', 'Wasco'],
    "REI"       : [8.3, 6.4,3.4,3.8,2.6,11.6,1.2,2.5,1.6],
    "Mortality" : [210,180,130,170,130,210,120,150,140]
}

df = pd.DataFrame(data)
df = df.sort_values('REI').reset_index(drop = True)

# print(df)
# df.to_csv('./9K9/Question1.csv')
# df = pd.read_csv('./9K9/Question1.csv').drop(columns= 'Unnamed: 0')
print(df)

## (a)
## When we want to check relationship between datas, we can use pearson's test 
# to check the degree fo correlation.

##We first make some basic analysis
# 1. linear regression
# Mortality = a + b * REI
b = sum( (df['REI'] - df['REI'].mean() )*(df['Mortality'] - df['Mortality'].mean()) )/sum((df['REI']-df['REI'].mean())**2)
a = df['Mortality'].mean() - df['REI'].mean()*b
print(a)
print(b)
# plt.plot(data['Mortality'], data['REI'])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data['REI'], data['Mortality'], marker = 'o')
# ypoints = np.array(range(int(round(max(df['REI']), 0))+1)) * b + a
# ax.plot(ypoints)
# plt.title("Figure 1")
# plt.xlabel('REI')
# plt.ylabel('Mortality')
# plt.show()
# print(data['REI'])
# # In this part we get an image of linear regrasion and it shows a linear relationship
# exit()

### Find outliers
Mortality = df['Mortality'].tolist()

Q1 = np.percentile(Mortality, 25)
Q3 = np.percentile(Mortality, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier = []
for x in Mortality:
    if x > upper_bound or x < lower_bound:
        outlier.append(x)

print(f"{len(outlier)} outliers have been found in Mortality.")

REI = df['REI'].tolist()

Q1 = np.percentile(REI, 25)
Q3 = np.percentile(REI, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier = []
for x in REI:
    if x > upper_bound or x < lower_bound:
        outlier.append(x)

print(f"{len(outlier)} outliers have been found in REI.")

## There is no outliers in our data.
## The next step is to test the homoscedasity.

exp_v = np.array(REI)*b+a
Residuals = exp_v - Mortality

# ax = sns.scatterplot(Residuals)
# ax = plt.axhline(0)
# ax = plt.title('Residual Plot')
# plt.show()

## Based on the plot, we can say residuals are close to the zero line, and
## did not show trend in any mode. So we can say this data is homoscedasity.

print(f"Date is homoscedasity.")

## After finish test of homoscedasity, next test is monotonic.
## The eaziest way to identify monotonic is view it. So let's 
## see the plot of data.

# ax = plt.scatter(REI, Mortality)
# plt.show()
print(f"Our data is monotonic")

##Based on the scatterplot, we can say the data is monotonic.

# exit()

## Then, we cna do correlation test.
## If we want to choose Pearson's R test, our data have to fit
## the requires that:
##  1. Variables are continuous
##  2. Residuals are normally distributed
##  3. Data are linear Related
##  4. No outliers
##  5. Homoscedastic
## Before, we have show the requirments of 1 3 4 5. Now to show
## the to Residuals are normally distributed.

## chi-square test it a tool that we are using to test the 
## normal distribution. 
# set Null hypothesis: H0 = Data are normal distributed
# set Alternative hypothesis: Data is not sampled from a normal distribution.
# set Significant level of 5%

# data = np.abs(Residuals)
# mean = np.mean(data)
# svd = np.power(np.var(data), 0.5)
# size = len(data)
# bin_size = 3
# bin_values = np.linspace(min(data), max(data)+0.01, bin_size+1)
# freq = []
# for x in range(bin_size):
#     freq.append( len(data[(data >= bin_values[x]) & (data < bin_values[x+1])]) )

# cdf = norm.cdf(bin_values, loc = mean, scale = svd)
# norm_freq = []
# for x in range(bin_size):
#     norm_freq.append((cdf[x+1] - cdf[x]) * size)

# bin_prob = np.array(freq )/size
# norm_prob = []
# for x in range(bin_size):
#     if x == 0:
#         norm_prob.append(cdf[x+1])
#     elif x == bin_size-1:
#         norm_prob.append((1 - cdf[x]) ) 
#     else:
#         norm_prob.append((cdf[x+1] - cdf[x]))
    
# chi2_stat, p_value = chisquare(bin_prob, norm_prob)

# print(chi2_stat)
# print(p_value)

# if p_value > 0.05:
#     print("Fail to reject H0: Data likely follow a normal distribution.")
# else:
#     print("Reject H0: Data do not follow a normal distribution.")
    
## Thus, our Residuals are likely fit a normal distribution
## which, satified the requirment 2.

## Next, do Pearson's R test.

pearson_corr, p_value = pearsonr(df['REI'], df['Mortality'])

print()
print(f'The pearson correlation value is: {pearson_corr}')
print(f'The P-value of this test is: {p_value}')

'''
Null hypothesis(H0):The correlation coefficient were in fact zero
Alternative hypothesis(H1):The correlation coefficient were not zero
'''

if p_value > 0.05:
    print("They are not Related.")
else:
    print("They are Related.")

# r = round(pearson_corr, 2)  # R value
# n = len(df['REI'])        # lenth of data
# t = r/np.power((1-r*r)/(n-2), 0.5)

# print(f"Our T-value is: {t}")


## (b)
# 做线形回归
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

