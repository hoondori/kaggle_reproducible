import os
import pylab
import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
# %matplotlib inline

train = pd.read_csv("./data/train.csv")

# Class imbalance is something to be aware of when training.
sns.countplot(x='Response', data=train, order=range(1, 9))


# There's a lot of missing data. Let's take a look
# First, let's find the columns with missing data. As you can see, only 1% of the rows for Medical_History_10 have values
print("train: nrows: %d, ncols: %d" % train.shape)
print("%20s \tCount \tPct missing" % 'Feature')
for column_name, column in train.transpose().iterrows():
    naCount = sum(column.isnull())
    if naCount > 0:
        print("%20s, \t%5d \t%2.2f%%" % (column_name, naCount, 100 * naCount / float(train.shape[0])))

# Plots for those variables with large amounts of missing data
fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,4))
train["Employment_Info_4"].plot(kind='hist',bins=20,xlim=(0,1),ax=axis1)
axis1.set_xlabel("Employment_Info_4")
axis1.set_ylabel("Count")
train["Employment_Info_6"].plot(kind='hist',bins=50,xlim=(0,1),ax=axis2)
axis2.set_xlabel("Employment_Info_6")
axis2.set_ylabel("Count")

# Insurance_History_5 has a few large outliers,
# otherwise most of the data is is less than 0.02.
# The data appears to be quantized
# Is there anything to learn in the data quantization for Insurance_History_5?
x = min(train["Insurance_History_5"][train["Insurance_History_5"]>0])
print("Min value >0 : %e, 1/(Min value>0) %f" % (x,1./x))

# List all of the Insurance_History_5 values greater than 0.02  (Max is 1.0)
print train["Insurance_History_5"][train["Insurance_History_5"]>0.02]

# Plot distribution for Insurance_History_5 with two different x-axis scale
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
train["Insurance_History_5"].plot(kind='hist',bins=20,xlim=(0,1),ax=axis1)
axis1.set_xlabel("Insurance_History_5")
axis1.set_ylabel("Count")
train["Insurance_History_5"][train["Insurance_History_5"]<0.034]\
    .plot(kind='hist',bins=60,xlim=(0,.05),ax=axis2)
axis1.set_xlabel("Insurance_History_5 (scaled X)")
axis1.set_ylabel("Count")

# Plot distribution for Family_Hist_x
pylab.rcParams['figure.figsize'] = (12.0,8.0)
fig, axisArr = plt.subplots(2,2)
train['Family_Hist_2'].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[0,0])
axisArr[0,0].set_xlabel("Family_Hist_2")
axisArr[0,0].set_ylabel("Count")
train["Family_Hist_3"].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[0,1])
axisArr[0,1].set_xlabel("Family_Hist_3")
axisArr[0,1].set_ylabel("Count")
train["Family_Hist_4"].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[1,0])
axisArr[1,0].set_xlabel("Family_Hist_4")
axisArr[1,0].set_ylabel("Count")
train["Family_Hist_5"].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[1,1])
axisArr[1,1].set_xlabel("Family_Hist_5")
axisArr[1,1].set_ylabel("Count")

# Worth noting: Medical History features
# with missing values aren't normalized. The max value is 240.
plt.clf()
train["Medical_History_1"].plot(kind='hist',xlim=(0,250),bins=100)
plt.xlabel("Medical_History_1")
plt.ylabel("Count")


# Plot distributions for Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
pylab.rcParams['figure.figsize'] = (12.0, 8.0)
fig, axisArr = plt.subplots(2,2)
train["Medical_History_10"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[0,0])
axisArr[0,0].set_xlabel("Medical_History_10")
axisArr[0,0].set_ylabel("Count")
train["Medical_History_15"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[0,1])
axisArr[0,1].set_xlabel("Medical_History_15")
axisArr[0,1].set_ylabel("Count")
train["Medical_History_24"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[1,0])
axisArr[1,0].set_xlabel("Medical_History_24")
axisArr[1,0].set_ylabel("Count")
train["Medical_History_32"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[1,1])
axisArr[1,1].set_xlabel("Medical_History_32")
axisArr[1,1].set_ylabel("Count")

# Product info for each response
pylab.rcParams['figure.figsize'] = (10.0, 14.0)
f, axisArr = plt.subplots(4,2)
for r in range(1,9):
    axs = axisArr[int((r-1)/2),(r-1)%2]
    sns.countplot(x='Product_Info_2',data=train[train["Response"]==r],
        order=['A1','A2','A3','A4','A5','A6','A7','A8',
                     'B1','B2',
                     'C1','C2','C3','C4',
                     'D1','D2','D3','D4',
                     'E1'],ax=axs)
    axs.set_ylabel('Count')
    axs.set_xlabel('Response: ' +str(r))

# Age vs Response
pylab.rcParams['figure.figsize'] = (10.0, 14.0)
f, axisarr = plt.subplots(4, 2)
for r in range(1,9):
    axs = axisarr[int((r-1)/2),(r-1)%2]
    train["Ins_Age"][train["Response"]==r].plot(kind='hist',bins=50,xlim=(0,1),ax=axs)
    axs.set_ylabel('Count')
    axs.set_xlabel('Response: '+str(r))



plt.show()

