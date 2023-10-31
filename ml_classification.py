# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:57:20 2023

@author: ayush
"""

import pandas as pd
import numpy as np

training_data = pd.read_csv("E:\MyProjects_2023\ML_Udemy\ml_classification (1)\storepurchasedata.csv")

x= training_data.iloc[:,:-1].values
y= training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =.20,random_state=0)