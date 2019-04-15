#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:56:40 2019

@author: kaptue
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from biokit.viz import corrplot

work_dir = "/home/kaptue/Bureau/Pro/4_dtu_compute_S2/computational data analysis/case 2/danske_web_data/"
file_name = "page_sep.csv"
file_path = work_dir + file_name


## a look at the data
sample_size = 1e4
page_df = pd.read_csv(file_path, nrows=sample_size, error_bad_lines=False, encoding='latin-1')
# already loaded
columns_names = page_df.columns
print(columns_names)
nan_columns_dic = {}
for col in columns_names:
   nan_columns_dic[col] = page_df[col].isna().sum()

## Then we can drop the column "referringpageinstanceid" because we have a lot of nan values and according to the
## the documentation it possesses the same information as the variable "pageinstanceid"
page_df_cleaned = page_df.drop(columns="referringpageinstanceid")

def read_data(data_path):
    return pd.read_csv(data_path, nrows=sample_size, error_bad_lines=False, encoding='latin-1')


def delete_redundant_column(col_name, dataframe):
    return dataframe.drop(columns=col_name)

## Let's now explore all the variables
    
### correlation structure between the categorical variables
corr = page_df_cleaned.loc[:,page_df_cleaned.dtypes == 'int64'].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
plt.show()

c = corrplot.Corrplot(page_df_cleaned.loc[:,page_df_cleaned.dtypes == 'int64'])
#c.plot()

### Data exploration
print(page_df_cleaned['iscustomer'].describe())

### number of url domain
page_location_list = np.unique(page_df_cleaned['pagelocationdomain'])
sns.countplot(y="pagelocationdomain", hue="iscustomer", data=page_df_cleaned)
plt.show()
page_df_cleaned['pagelocationdomain'].value_counts().plot(kind='bar')
plt.show()

### let's check the observations with the page location "https://www.danskebank.dk"
check_df = pd.DataFrame(page_df_cleaned[page_df_cleaned['pagelocationdomain']==page_location_list[-1]])

### Now we can have a look at the full page location
print(page_df_cleaned['pagelocation'].describe())
page_location_occurences = page_df_cleaned['pagelocation'].value_counts()

### difference between pagesequenceinsession and pagesequenceinattribution
diff_vector = page_df_cleaned["pagesequenceinsession"].sub(page_df_cleaned["pagesequenceinattribution"], axis = 0)
print("Number difference between the two columns :", len(diff_vector[diff_vector != 0]))
page_sequences_in_sesion_list = np.unique(page_df_cleaned["pagesequenceinsession"])
page_sequences_in_attribution_list = np.unique(page_df_cleaned["pagesequenceinattribution"])



