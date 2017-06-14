# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:19:41 2017

@author: Ashish
"""

import pandas as pandas
import statsmodels.formula.api as smf
import numpy
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api as sm

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#load csv file
data = pandas.read_csv('GapMinder_mine.csv', low_memory=False)

#Replace all blank spaces in file with NAN
data = data.apply(lambda x: x.str.strip()).replace('', numpy.nan)

# listwise deletion of missing values
data = data.dropna()

# convert variables to numeric format using convert_objects function
data['lifeexpectancy'] = pandas.to_numeric(data['lifeexpectancy'], errors='coerce')
data['urbanrate'] = pandas.to_numeric(data['urbanrate'], errors='coerce')
data['co2emissions'] = pandas.to_numeric(data['co2emissions'], errors='coerce')
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors='coerce')


#centered Explanatory variables
data['urbanrate_c']=(data['urbanrate'] - data['urbanrate'].mean())
data['co2emissions_c']=(data['co2emissions'] - data['co2emissions'].mean())
data['incomeperperson_c']=(data['incomeperperson'] - data['incomeperperson'].mean())

#Multiple Regression (Life Expectancy and All other)
reg1 = smf.ols('lifeexpectancy ~ urbanrate_c + co2emissions_c + incomeperperson_c + I(incomeperperson_c**2)', data=data).fit()
print (reg1.summary())


reg2 = smf.ols('lifeexpectancy ~ urbanrate_c', data=data).fit()
print (reg1.summary())


#Polynimial Regression
reg3 = smf.ols('lifeexpectancy ~ urbanrate_c + I(urbanrate_c**2)', data=data).fit()
print (reg2.summary())

scat1 = seaborn.regplot(x="urbanrate", y="lifeexpectancy", scatter=True, order=2, data=data)
plt.xlabel('urbanrate ')
plt.ylabel('lifeexpectancy')



#linear Regression (Life Expectancy and Income Per Person)
reg4 = smf.ols('lifeexpectancy ~ incomeperperson', data=data).fit()
print (reg2.summary())

#Polynimial Regression (Life Expectancy and Income Per Person)
reg5 = smf.ols('lifeexpectancy ~ incomeperperson_c + I(incomeperperson_c**2)', data=data).fit()
print (reg3.summary())




#scatterplot between Life Expectancy and Income Per Person
scat1 = seaborn.regplot(x="incomeperperson", y="lifeexpectancy", scatter=True, order=2, data=data)
plt.xlabel('incomeperperson ')
plt.ylabel('lifeexpectancy')


#Q-Q plot for normality
fig1=sm.qqplot(reg5.resid, line='r')


# simple plot of residuals
stdres=pandas.DataFrame(reg5.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')

# additional regression diagnostic plots
fig2 = plt.figure(figsize=(12,8))
fig2 = sm.graphics.plot_regress_exog(reg5,  "incomeperperson_c", fig=fig2)

# leverage plot
fig3=sm.graphics.influence_plot(reg5, size=8)
print(fig3)


