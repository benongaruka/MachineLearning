# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 21:56:31 2018

@author: Benon
"""
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Importing data from csv
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#Inspect data
train_df.head()
test_df.head()
train_df.sample(3) #Sample of three random rows
#Column names
print(train_df.columns.values)

#Preprocessing
#Drop unnecessary columns
train_df = train_df.drop(["PassengerId", "Name", "Ticket"], axis=1)
test_df = test_df.drop(["PassengerId", "Name", "Ticket"], axis=1)

#This will help you see what columns have null values
train_df.info()
print("-"*40)
test_df.info()

#Combine the datasets for analysis
full = train_df.append(test_df, ignore_index=True)
print("Datasets:", "full:", full.shape, "train:",train_df.shape)

#Numerical summary
full.describe()

#Heat map to get a sense of which variables are important
corr = full.corr() #Gives us matrix of correlation variables
_, ax = plt.subplots(figsize=(12,10))
cmap = sns.diverging_palette(220,10, as_cmap = True)




#Take care of missing values
train_df.Embarked = train_df.Embarked.fillna("Unknown")
train_df.Embarked.values


#Visualizing the Data
#https://seaborn.pydata.org/examples/
#Barplot
sns.barplot(x="Embarked", y="Survived", hue="Sex", data = train_df)

#lm plot. scatter plot that fits a regression line 
sns.set(style="ticks")

df = sns.load_dataset("anscombe")

sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df, col_wrap=2, ci=None, palette="muted",size=4, scatter_kws={"s":50, "alpha":1})

sns.pointplot(x="Pclass", y="Survived", hue="Sex",data=train_df, palette={"male":"blue","female":"pink"}, markers=['*','o'], linestyles=['-','--'])


#Playing with aesthetics
#Plot for generating 7 offset plots
def sineplot(flip=1):
    x = np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*.5)*(7-i)*flip)
        
sineplot()

#Switch to seaborn defaults
sns.set()
sineplot()

#Two sets of funtions to manipulate plot parameters
#1. Style: axes_style(), set_style() 2. Scale: plotting_context(), set_context()

#Figure styles
#Grids on the plot
sns.set_style("whitegrid")
data = np.random.normal(size=(20,6)) + np.arange(6)
sns.boxplot(data=data)
sineplot()

#non grid themes, dark, white, ticks
sns.set_style("dark")
sineplot()

sns.set_style("white")
sineplot()

#Remove to and right spine
sineplot()
sns.despine() #This takes args if you wish to determine which spines to remove, e.g. sns.despine(left=True)

#Offsetting spines
f, ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True) #trim adjusts the range of the axis when they don't cover the whole range of the axis

#Switch back and forth between styles using axes_style()
with sns.axes_style("darkgrid"):
    plt.subplot(211) #211=nrow=2,ncol=1,plot_number=1
    sineplot()
plt.subplot(212)
sineplot(-1)