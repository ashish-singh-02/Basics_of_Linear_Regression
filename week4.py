# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 23:02:58 2017

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

mean_income = data['incomeperperson'].mean()
print(mean_income)

mean_life = data['lifeexpectancy'].mean()
print(mean_life)

def create_New_Varable_For_Income(income):
   if income <=mean_income:
       return 0
   elif income > mean_income:
       return 1

def create_New_Varable_For_Life(life):
   if life <=mean_life:
       return 0
   elif life > mean_life:
       return 1


def create_New_Varable_For_Urban(urban):
   if (urban >0) & (urban <=50):
       return 0
   elif(urban >50) & (urban <=100):
       return 1


data['incomeperperson_new']= data['incomeperperson'].apply(create_New_Varable_For_Income)
data['urbanrate_new']= data['urbanrate'].apply(create_New_Varable_For_Urban)
data['life_new']= data['lifeexpectancy'].apply(create_New_Varable_For_Life)
   
# logistic regression
lreg1 = smf.logit(formula = 'life_new ~ incomeperperson_new', data = data).fit()
print (lreg1.summary())
# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))
    
# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))



# logistic regression with Urban Rate and Income Per Person
lreg2 = smf.logit(formula = 'life_new ~ incomeperperson_new + urbanrate_new', data = data).fit()
print (lreg2.summary())

