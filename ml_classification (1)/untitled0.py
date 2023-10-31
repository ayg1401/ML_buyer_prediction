# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 19:19:56 2023

@author: ayush
"""

import pickle

import numpy as np
local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scalar = pickle.load(open('sc.pickle','rb'))
new_pred = local_classifier.predict(local_scalar.transform(np.array([[20,75000]])))
new_pred_prob = local_classifier.predict_proba(local_scalar.transform(np.array([[20,75000]])))