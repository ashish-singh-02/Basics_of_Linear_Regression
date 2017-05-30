# -*- coding: utf-8 -*-
"""
Created on Sun May 28 22:15:30 2017

@author: Ashish
"""

import numpy as numpyp
import pandas as pandas
import statsmodels.api
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#load csv file
data = pandas.read_csv('GapMinder_mine.csv', low_memory=False)

# convert variables to numeric format using convert_objects function
data['lifeexpectancy'] = pandas.to_numeric(data['lifeexpectancy'], errors='coerce')
data['urbanrate'] = pandas.to_numeric(data['urbanrate'], errors='coerce')
data['co2emissions'] = pandas.to_numeric(data['co2emissions'], errors='coerce')




############################################################################################
# This is how we would have centered Explanatory variables in case of multiple explanatory variables
############################################################################################
mean_urban = data['co2emissions'].mean()
data['urbanrate1'] = data['co2emissions']  - mean_urban

print("Mean After Centering the Exp Variable :", data['co2emissions'].mean())
print("Mean Before Centering the Exp Variable :", data['co2emissions'].mean())

print ("==============================================================================")

############################################################################################
# BASIC LINEAR REGRESSION
############################################################################################
scat1 = seaborn.regplot(x="urbanrate", y="lifeexpectancy", scatter=True, data=data)
plt.xlabel('Urban Rate')
plt.ylabel('Life Expectancy')
plt.title ('Scatterplot for the Association Between Urban Rate and Life Expectancy')
print(scat1)

print ("OLS regression model for the association between urban rate and Life Expectancy")
reg1 = smf.ols('lifeexpectancy ~ urbanrate', data=data).fit()
print (reg1.summary())


