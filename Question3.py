import numpy as np
import pandas as pd
import math
from scipy.stats import *
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data_1 = {
    "Disc #1": [55, 72, 26, 65, 104, 62, 89, 52, 76, 46, 100, 72, 35]}
data_2 = {
    "Disc #2": [71, 62, 37, 46, 36,  44, 63, 71, 49, 44, 41,  76, 55, 64, 41]
}
df_1 = pd.DataFrame(data_1)
df_2 = pd.DataFrame(data_2)

print(df_1.describe())
print(df_2.describe())
print()

## F-test
Var_1 = df_1.var()[0]
Var_2 = df_2.var()[0]

if Var_1 > Var_2:
    F = Var_1 / Var_2
    dfn, dfd = len(df_1) - 1, len(df_2) - 1
else: 
    F = Var_2 / Var_1
    dfn, dfd = len(df_2) - 1, len(df_1) - 1
    
p_value = 1 - f.cdf(F, dfn, dfd)

print('=========F-test=========')
print(f"Variance of Disc #1: {Var_1:.4f}")
print(f"Variance of Disc #2: {Var_2:.4f}")
print(f"F_value: {F:.4f}")
print(f"p_value: {p_value:.4f}")

if p_value < 0.05:
    print("Rejct null hypothesis: Two data have different Variance.")
else:
    print("Fail to rejct null hypothesis: Two data have same Variance.")
    

print("\n")
print("=========T-test=========")
T, p_value = ttest_ind(df_1['Disc #1'].to_list(), df_2['Disc #2'].to_list(), equal_var=False)
print(f"T_value: {T:.4f}")
print(f"p_value: {p_value:.4f}")

if p_value < 0.05:
    print("Rejct null hypothesis: Two data have different Mean value.")
else:
    print("Fail to rejct null hypothesis: Two data have same Mean value.")
    
print("\n")

## Outlier check
data1 = [55, 72, 26, 65, 104, 62, 89, 52, 76, 46, 100, 72, 35]
data2 = [71, 62, 37, 46, 36, 44, 63, 71, 49, 44, 41, 76, 55, 64, 41]


def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = []
    for x in data:
        if x < lower_bound or x > upper_bound:
            outliers.append(x)
    return outliers, lower_bound, upper_bound

outliers1, lb1, ub1 = detect_outliers_iqr(data1)
outliers2, lb2, ub2 = detect_outliers_iqr(data2)

print("Disc #1 Outliers:", outliers1)
print(f"Disc #1 range: [{lb1:.2f}, {ub1:.2f}]")
print("Disc #2 Outliers:", outliers2)
print(f"Disc #2 range: [{lb2:.2f}, {ub2:.2f}]")
print()

## (ii)
data = {
    "Month": [
        "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01",
        "2022-06-01", "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01",
        "2022-11-01", "2022-12-01", "2023-01-01", "2023-02-01", "2023-03-01",
        "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01",
        "2023-09-01", "2023-10-01", "2023-11-01", "2023-12-01"
    ],
    "Social Media Engagement": [
        4174, 4507, 1860, 2294, 2130, 2095, 4772, 4092, 2638, 3169, 
        1466, 2238, 1330, 2482, 3135, 4444, 4171, 3919, 4735, 1130, 
        2685, 4380, 1769, 3391
    ],
    "E-commerce Sales (GBP)": [
        43733.69, 40867.71, 18153.32, 21248.07, 18954.22, 21034.44, 45208.41, 
        40848.7, 26005.26, 30055.2, 13297.71, 21862.1, 13369.99, 24975.03, 
        31345.29, 45607.35, 44238.92, 37214.39, 48760.05, 12248.26, 
        28668.32, 46515.17, 17518.32, 35302.87
    ]
}

# Load into a DataFrame
df = pd.DataFrame(data)

print(df)

fig = plt.figure()
ax = fig.add_subplot(111)
ax = plt.plot(df['Social Media Engagement'],df['E-commerce Sales (GBP)'],'o')
plt.xlabel('Social Media Engagement')
plt.ylabel('E-commerce Sales (GBP)')
plt.title('Engagement v.s. Sales')


## correlation test pearson's r
SME = df['Social Media Engagement']
Sales = df['E-commerce Sales (GBP)']

corr, p_value = pearsonr(SME, Sales)


print()
print(f'The pearson correlation value is: {corr:.4f}')
print(f'The P-value of this test is: {p_value:.4f}')
print()

import statsmodels.api as sm

X = sm.add_constant(SME.tolist())
y = Sales.tolist()

model = sm.OLS(y, X).fit()

print(model.summary())
print()

plt.show()