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

def cat_onehot(df):
    """One hot encodes all categorical variables from a DataFrame. 
       - Cretes a new DataFrame with columns names = orginal categorical name + unqiue value (i.e engine_v6 or color_red)
       - Drops original categorical column 
    Args:
        df (pandas DataFrame): feature variable data frame
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelBinarizer
    enc = OneHotEncoder(handle_unknown = "ignore", sparse='False')
    label_enc = LabelBinarizer()

    for c in df.select_dtypes(exclude=['number']):
        # get matrix and names of new features
        mat = enc.fit_transform(df[[c]])
        names = enc.get_feature_names_out()
        label_enc.fit(df[c])
        # transform encoded 
        transformed = label_enc.transform(df[c]) 
        # create dataframe of new features with names as columns
        ohe_df = pd.DataFrame(transformed, columns=names)
        df = pd.concat([df, ohe_df], axis=1).drop([c], axis=1)
        
    return df

# Automating backward elimination technique

def DoBackwardElimination(the_regressor, X, y, X_val, minP2eliminate):
    """_summary_

    Args:
        the_regressor (_type_): _description_
        X (DataFrame): _description_
        y (Series): _description_
        X_val (DataFrame): _description_
        minP2eliminate (float): _description_

    Returns:
        _type_: _description_
    """
    
    assert np.shape(X)[0] == np.shape(y)[0], 'Length of X and y do not match'
    assert minP2eliminate > 0, 'Minimum P value to eliminate cannot be zero or negative'
    
    original_list = list(range(0, np.shape(the_regressor.pvalues)[0]))
    
    max_p = 10        # Initializing with random value of maximum P value
    i = 0
    r2adjusted = []   # Will store R Square adjusted value for each loop
    r2 = []           # Will store R Square value  for each loop
    list_of_originallist = [] # Will store modified index of X at each loop
    classifiers_list = [] # fitted classifiers at each loop
    
    while max_p >= minP2eliminate:
        
        p_values = list(the_regressor.pvalues)
        r2adjusted.append(the_regressor.rsquared_adj)
        r2.append(the_regressor.rsquared)
        list_of_originallist.append(original_list)
        
        max_p = max(p_values)
        max_p_idx = p_values.index(max_p)
                
        if max_p < minP2eliminate:
            
            print('Max P value found less than ', str(minP2eliminate), ' without 0 index...Loop Ends!!')
            
            break
        
        val_at_idx = original_list[max_p_idx]
        
        idx_in_org_lst = original_list.index(val_at_idx)
        
        original_list.remove(val_at_idx)
        
        print('Popped column index out of original array is {} with P-Value {}'.format(val_at_idx, np.round(np.array(p_values)[max_p_idx], decimals= 4)))
        
        X_new = X.iloc[:,original_list]
        X_val_new = X_val.iloc[:,original_list]
        
        the_regressor = smf.OLS(endog = y, exog = X_new).fit()
        classifiers_list.append(the_regressor)
        
        print('==================================================================================================')
        
    return classifiers_list, r2, r2adjusted, list_of_originallist, X_new, X_val_new

def Calculate_Error(original_values, predicted_values):
    assert len(original_values) == len(predicted_values), 'Both list should have same length'
    temp = 0
    error = 0
    n = len(original_values)
    for o, p in zip(original_values, predicted_values):
        temp = temp + ((o-p)**2)
        
    temp = temp/n
    error = np.sqrt(temp)
    return error

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Importing training and test sets
DATA_PATH = 'C:\\Users\\tawate\OneDrive - SAS\\01_Training\\08_Kaggle\\Merc_Benz_Greener_Manufacturing\\Data\\'
TRAIN = 'train.csv'
TEST = 'test.csv'
train_df = pd.read_csv(DATA_PATH + TRAIN)
test_df = pd.read_csv(DATA_PATH + TEST)

""" 
    Initial Data Exploration
"""
train_df.info
train_df.dtypes
train_df.describe()

# Extract target, id, and feature variables
train_id = train_df['ID']
train_tg = train_df['y']
train_var = train_df.iloc[:,2:]
test_var = test_df.iloc[:,2:]

# Feature variable exploration
train_var.info()
var_types = train_var.dtypes
train_var.shape
var_ex = train_var.describe()
var_num = train_var.select_dtypes(include=['number'])
var_cat = train_var.select_dtypes(exclude=['number'])

# Feature variable missing values
    # No missing values!
missing_vals = train_var.isnull().sum()
tg_missing_vals = train_tg.isnull().sum()

# Historgram for each numeric feature
for var in var_num:
    var_num[var].plot.hist(grid = True, color = '#607c8e')

# Unique values for each numeric and categorical features
for col in var_num:
    print(col)
    print(var_num[col].nunique())
    
for col in var_cat:
    print(col)
    print(var_cat[col].nunique())
    print(var_cat[col].value_counts())
    
# Target variable exploration
train_tg.plot.hist(grid = True, color = '#607c8e')
tg_stats = train_tg.describe()

# Coding Categorical Variables
train_hot = cat_onehot(train_var)
test_hot = cat_onehot(test_var)
x_train, x_valid, y_train, y_valid = train_test_split(train_hot,train_tg, test_size = .3, 
                                                    random_state=0)

""" Dimensionality Reduction
        Feature removal methods:
            - Backward Elimination
            - Forward Elimination
            - Random forest (variable importance)
"""

# Regression Analysis without Feature Elimination
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import statsmodels.regression.linear_model as sm
import seaborn as sns
import statsmodels.regression.linear_model as smf
from sklearn.preprocessing import PolynomialFeatures
import random

regressor_OLS = smf.OLS(endog = y_train, exog = x_train).fit()
print(regressor_OLS.summary())

# Residual Graphs w.o elimination
sns.set(style='ticks')
plt.figure(figsize=(14,8))
y_pred_train = regressor_OLS.predict(x_train)
y_pred_test = regressor_OLS.predict(x_valid)

plt.scatter(y_train, y_train-y_pred_train, s= 2, c= 'Red', alpha=0.8)
plt.scatter(y_valid, y_valid-y_pred_test, s= 2, c= 'Blue', alpha=0.8)
plt.plot([0, 18000], [0,0], '-k', linewidth = 3, alpha = 0.3)
p = plt.xlim((0, 200))
p = plt.ylim((-100, 100))
p = plt.legend(['Reference Line','Training Data Residual', 'Testing Data Residual'])
p = plt.title('Residual Graph')


train_data_error = Calculate_Error(original_values=y_train, predicted_values=y_pred_train)
test_data_error = Calculate_Error(original_values=y_valid, predicted_values=y_pred_test)

print('MSE for training data is {}'.format(np.round(train_data_error, 4)))
print('MSE for testing data is {}'.format(np.round(test_data_error, 4)))

# Plotting Original vs Predicted Values
plt.figure(figsize=(50, 30))
plt.subplot(2,1,1)
# plt.figure(figsize=(15,10))
p = plt.plot(range(0, len(y_train)), y_train, color = 'red')
p = plt.plot(range(0, len(y_train)), y_pred_train, color = 'blue')
plt.title(r'$ \mathrm{\mathsf{Training Data Output}}$')
plt.legend(['Original Output', 'Predicted Output'])
plt.xlabel(r'$Observation  Number \longrightarrow$')
plt.ylabel(r'$Output \longrightarrow$')
plt.ylim(-10, 200)


plt.subplot(2,1,2)
p = plt.plot(range(0, len(y_valid)), y_valid, color = 'red')
p = plt.plot(range(0, len(y_valid)), y_pred_test, color = 'blue')
plt.title(r'$ \mathrm{\mathsf{Testing Data Output}}$')
plt.legend(['Original Output', 'Predicted Output'])
plt.xlabel(r'$Observation  Number \longrightarrow$')
plt.ylabel(r'$Output \longrightarrow$')
plt.ylim(-10, 200)

plt.subplots_adjust(hspace=0.4)

# Residual Plots
training_residual = []
for o, p in zip(y_train, y_pred_train):
    training_residual.append(o-p)
    
testing_residual = []
for o, p in zip(y_valid, y_pred_test):
    testing_residual.append(o-p)


plt.figure(figsize=(15,10))

plt.subplot(2,1,1)
p = plt.scatter(list(range(0, len(y_train))),training_residual)
p = plt.plot([0, 50], [0, 0], '-k')
plt.text(30,100, 'Mean of residual is {}'.format(np.round(np.mean(training_residual), 4)))
plt.ylim(-50, 50)
plt.xlim(0,50)
plt.title('Training Residuals')
plt.xlabel(r'$Observation  Number \longrightarrow$')
plt.ylabel(r'$Residual \longrightarrow$')


plt.subplot(2,1,2)
p = plt.scatter(list(range(0, len(y_valid))),testing_residual)
p = plt.plot([0, 50], [0, 0], '-k')
plt.text(30,100, 'Mean of residual is {}'.format(np.round(np.mean(testing_residual), 4)))
plt.ylim(-50, 50)
plt.xlim(0,50)
plt.title('Testing Residuals')
plt.xlabel(r'$Observation  Number \longrightarrow$')
plt.ylabel(r'$Residual \longrightarrow$')

plt.subplots_adjust(hspace= 0.4)

print('Mean of \'price\' for training data is {}.'.format(np.round(np.mean(y_train),4)))
print('Mean of residual of \'price\' for training data is {}.'.format(np.round(np.mean(training_residual),4)))

print('Mean of \'price\' for testing data is {}.'.format(np.round(np.mean(y_valid),4)))
print('Mean of residual of \'price\' for training data is {}.'.format(np.round(np.mean(testing_residual),4)))

# Backward Elimination (https://www.kaggle.com/code/ashishsaxena2209/step-by-step-regression-backward-elimination/notebook)
regressor_list, r2, r2adjusted, list_of_changes, x_train_new, x_valid_new = DoBackwardElimination(the_regressor=regressor_OLS, 
                                                                                                  X= x_train, 
                                                                                                  y= y_train, 
                                                                                                  X_val = x_valid, 
                                                                                                  minP2eliminate = 0.05)



# Random Forest Selection (https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
rf = RandomForestRegressor(random_state=0)
rf_orig = rf.fit(x_train, y_train)
feature_importance = rf_orig.feature_importances_
sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
sel.fit(x_train, y_train)
selected_feat= x_train.columns[(sel.get_support())]
len(selected_feat)
x_train_rf = x_train[selected_feat]
x_valid_rf = x_valid[selected_feat]

# Model diagnostics after selection
#-------------Linear Regression-------------
reg_back = smf.OLS(endog = y_train, exog = x_train_new).fit()
reg_rf = smf.OLS(endog = y_train, exog = x_train_rf).fit()
print(regressor_OLS.summary())
print(reg_rf.summary())
print(reg_back.summary())

# Back Selection
reg_back_pred = reg_back.predict(x_valid_new)
reg_back_er = abs(reg_back_pred - y_valid)
# MAE
print('Mean abs error: ', round(np.mean(reg_back_er), 2), 'secs')
# MAPE
reg_back_mape = 100* (reg_back_er / y_valid)
# Accuracy
rf_reg_acc = 100 - np.mean(reg_back_mape)
print('Accuracy:', round(rf_reg_acc, 2), '%.')

# RF Variable Selection
reg_rf_pred = reg_rf.predict(x_valid_rf)
reg_rf_er = abs(reg_rf_pred - y_valid)
# MAE
print('Mean abs error: ', round(np.mean(reg_rf_er), 2), 'secs')
# MAPE
reg_rf_mape = 100* (reg_rf_er / y_valid)
# Accuracy
rf_reg_acc = 100 - np.mean(reg_rf_mape)
print('Accuracy:', round(rf_reg_acc, 2), '%.')

#------------Random Forest----------------
# Back Selection
rf_back = rf.fit(x_train_new, y_train)
rf_back_pred = rf_back.predict(x_valid_new)
rf_back_er = abs(rf_back_pred - y_valid)
# MAE
print('Mean abs error: ', round(np.mean(rf_back_er), 2), 'secs')
# MAPE
rf_back_mape = 100* (rf_back_er / y_valid)
# Accuracy
rf_back_acc = 100 - np.mean(rf_back_mape)
print('Accuracy:', round(rf_back_acc, 2), '%.')

# RF Variable Selection
rf_rf = rf.fit(x_train_rf, y_train)
rf_rf_pred = rf_rf.predict(x_valid_rf)
rf_rf_er = abs(rf_rf_pred - y_valid)
# MAE
print('Mean abs error: ', round(np.mean(rf_rf_er), 2), 'secs')
# MAPE
rf_rf_mape = 100* (rf_rf_er / y_valid)
# Accuracy
rf_rf_acc = 100 - np.mean(rf_rf_mape)
print('Accuracy:', round(rf_rf_acc, 2), '%.')

#---------Ridge Regression-----------------
from sklearn.linear_model import Ridge
# Back Selection
alpha = 10
n, m = x_train_new.shape
I = np.indentity(m)
w = np.dot(np.dot(np.linalg.inv(np.dot(x_train_new.T, x_train_new) + alpha*I), x_train_new.T),y_train)
rr_back = Ridge(alpha = 10)
rr_back.fit(x_train_new, y_train)
w = rr_back.coef_


#-------Test Output-----------------------
x_test_rf = test_hot[selected_feat]
rf_test_pred = rf_rf.predict(x_test_rf)


