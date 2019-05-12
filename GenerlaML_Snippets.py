# -*- coding: utf-8 -*-
"""
Created on Sun Jan 07 22:33:11 2018
#Functions for data manipulatin and visualization, as well as ml
@author: Benon
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
