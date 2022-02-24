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
train_var = train_df[2:]