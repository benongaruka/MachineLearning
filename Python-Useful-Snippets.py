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

#Visualizing the Data
sns.barplot(x="Embarked", y="Survived", hue="Sex", data = train_df)
