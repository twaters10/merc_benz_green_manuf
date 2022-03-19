""" Project Description:
Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. 
These include, for example, the passenger safety cell with crumple zone, the airbag and intelligent assistance systems. 
Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium car makers. 
Daimler’s Mercedes-Benz cars are leaders in the premium car industry. With a huge selection of features and options, 
customers can choose the customized Mercedes-Benz of their dreams. .

To ensure the safety and reliability of each and every unique car configuration before they hit the road, 
Daimler’s engineers have developed a robust testing system. But, optimizing the speed of their testing system for 
so many possible feature combinations is complex and time-consuming without a powerful algorithmic approach. As one of the 
world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Daimler’s production lines.

In this competition, Daimler is challenging Kagglers to tackle the curse of dimensionality and reduce the time that 
cars spend on the test bench. Competitors will work with a dataset representing different permutations of Mercedes-Benz 
car features to predict the time it takes to pass testing. Winning algorithms will contribute to speedier testing, resulting 
in lower carbon dioxide emissions without reducing Daimler’s standards.

Data Set Description:
This dataset contains an anonymized set of variables, each representing a custom feature in a Mercedes car. For example, 
a variable could be 4WD, added air suspension, or a head-up display.

The ground truth is labeled ‘y’ and represents the time (in seconds) that the car took to pass testing for each variable.

train.csv --> the training set
test.csv --> the test set
sample_submission.csv --> a sample submission file in the forrect format
"""

import pandas as pd
import numpy as np

# Importing training and test sets
DATA_PATH = 'C:\\Users\\tawate\OneDrive - SAS\\01_Training\\08_Kaggle\\Merc_Benz_Greener_Manufacturing\\Data\\'
TRAIN = 'train.csv'
TEST = 'test.csv'
train_df = pd.read_csv(DATA_PATH + TRAIN)
test_df = pd.read_csv(DATA_PATH + TEST)

# Initial Data Exploration
train_df.info
train_df.dtypes
train_df.describe()

# Extract target, id, and feature variables
train_id = train_df['ID']
train_tg = train_df['y']
train_var = train_df.iloc[:,2:]

# Feature variable exploration
train_var.info()
var_types = train_var.dtypes
train_var.shape
var_ex = train_var.describe()
var_num = train_var.select_dtypes(include=['number'])
var_cat = train_var.select_dtypes(exclude=['number'])

# Historgram for each numeric feature
for var in var_num:
    var_num[var].plot.hist(grid = True, color = '#607c8e')

# Unique values for each numeric and categorical features
for col in var_num:
    print(var_num[col].nunique())
        
# Target variable exploration
train_tg.plot.hist(grind = True, color = '#607c8e')
tg_stats = train_tg.describe()

# Categorical Encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse='False')
# One hot Encode Example
enc_df_example = pd.DataFrame(enc.fit_transform(var_cat[['X3']]).toarray())

# One hot encode categorical columns with less than 10 unique values
# Determine method for columns with more than 10 unique values
for col in var_cat:
    if var_cat[col].nunique() > 10:
        print(var_cat[col].nunique())
    else:
        enc_df = pd.DataFrame(enc.fit_transform(var_cat[[col]]).toarray())
        var_num.append(enc_df)

# Feature engineering
from sklearn.preprocessing import StandardScaler