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
_ = sns.heatmap(corr, cmap = cmap, square = True, cbar_kws={"shrink" :.9}, ax=ax, annot=True, annot_kws={"fontsize":12})

#Age and Survival


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
