#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:47:00 2019

@author: kaptue
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import warnings
warnings.filterwarnings('ignore')


file_name = "page_aug.csv"
sample_size = 1e4
page_sample = pd.read_csv(file_name, nrows=sample_size, error_bad_lines=False, encoding='latin-1')
page_sample_cleaned = page_sample.drop(columns="referringpageinstanceid")
y = page_sample_cleaned["iscustomer"]
cont_names = list(["sessionnumber", "pageinstanceid", "eventtimestamp", "pagesequenceinsession"])
x_pagelocationdomain_dummies = pd.get_dummies(page_sample_cleaned["pagelocationdomain"])
# x_pagetitle_dummies = pd.get_dummies(page_sample_cleaned["pagetitle"])
x = pd.concat([page_sample_cleaned[cont_names], x_pagelocationdomain_dummies], axis=1)


#### Decison tree with the variable iscustomer as response variable
dtree=DecisionTreeClassifier()
# Casting response into categorical, as decision trees expects cat response
dtree.fit(x, y)
levels_list = list(np.unique(page_sample_cleaned["pagelocationdomain"]))
features_names = levels_list.extend(levels_list)

#### we export the tree as a graphviz format
dot_data = export_graphviz(dtree, out_file=None, 
                         feature_names=features_names,  
                         class_names='iscustomer',  
                         filled=True, rounded=True,  
                         special_characters=True)  


#with open("page_tree.dot", 'w') as f:
#    export_graphviz(dtree, out_file=f,
#                    feature_names=features_names)

graph_png = pydotplus.graph_from_dot_data(dot_data)
graph_png.write_pdf('page_tree.pdf')