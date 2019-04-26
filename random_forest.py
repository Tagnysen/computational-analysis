#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:15:45 2019

@author: kaptue
"""
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
import pydotplus


file_name = "page_aug.csv"
sample_size = 1e5
page_sample = pd.read_csv(file_name, nrows=sample_size, error_bad_lines=False, encoding='latin-1')
page_sample_cleaned = page_sample.drop(columns="referringpageinstanceid")
y = page_sample_cleaned["iscustomer"]
cont_names = list(["sessionnumber", "pageinstanceid", "eventtimestamp", "pagesequenceinsession"])
x_pagelocationdomain_dummies = pd.get_dummies(page_sample_cleaned["pagelocationdomain"])
# x_pagetitle_dummies = pd.get_dummies(page_sample_cleaned["pagetitle"])
x = pd.concat([page_sample_cleaned[cont_names], x_pagelocationdomain_dummies], axis=1)

[N, P] = x.shape
clf = RandomForestClassifier(bootstrap=True, oob_score=True, criterion='gini',random_state=0, max_depth=10)
clf.fit(x, y)
i = 1

# number of trees
n_estimators = [int(x) for x in np.linspace(100, 500, 2)]
# Try to add more of the parameters from the model and then add them to this dict to see how it affects the model.
param_grid = {
    'n_estimators': n_estimators,
}
rf_grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=2, iid=True, n_jobs=-1)

# Fit the grid search model
rf_grid.fit(x, y)

# # Look at one random forrest and the importance of the features
one_rf = RandomForestClassifier(bootstrap=True, oob_score=True, max_depth = 10, n_estimators=100,
                                criterion='gini', random_state=0)
score = one_rf.fit(x, y)
headers = ["name", "score"]
values = sorted(zip(range(0,P), one_rf.feature_importances_), key=lambda x: x[1] * -1)
# See which features are deemed most important by the classifier
print(tabulate(values, headers, tablefmt="plain"))
print ('Random Forest OOB error rate: {}'.format(1 - one_rf.oob_score_))

for tree_in_forest in one_rf.estimators_:
    dot_data = export_graphviz(tree_in_forest,
                out_file=None,        
                feature_names=x.columns,
                filled=True,
                rounded=True)
    graph_png = pydotplus.graph_from_dot_data(dot_data)
    graph_name = 'forest_tree_'+ str(i) + '.pdf'
    i += 1
    graph_png.write_pdf(graph_name)