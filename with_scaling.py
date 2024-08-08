# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:21:05 2024

@author: Bharti
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

# for chi square test 
from scipy.stats import chi2_contingency

external_df = pd.read_excel('D:/Pinnacle_WorkSpace/CSC_SPV/CIBIL_DATA/External_Cibil_Dataset.xlsx')
internal_df = pd.read_excel('D:/Pinnacle_WorkSpace/CSC_SPV/CIBIL_DATA/Internal_Bank_Dataset.xlsx')

external_df1 = external_df.copy()
internal_df1 = internal_df.copy()

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)

external_df_cols = external_df.columns
external_df_cols = set(external_df_cols)
internal_df_cols = internal_df.columns
internal_df_cols = set(internal_df_cols)
external_df.isnull().sum()
external_df.dtypes

cols_notin_internal = external_df_cols - internal_df_cols

# external_df.isna().sum()

internal_df = internal_df[internal_df['Age_Oldest_TL'] != -99999]
columns_to_be_removed = []
for col in external_df_cols:
    if external_df.loc[external_df[col] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(col)
        
        
external_df.drop(columns = columns_to_be_removed,inplace=True)

for col in external_df.columns:
    external_df = external_df[external_df[col] != -99999]

for col in internal_df.columns:
    in_null_vals = internal_df[internal_df[col] == -99999]

# internal_df.describe()
# external_df.describe()

common_col = []
for i in external_df.columns:
    if i in internal_df.columns:
        common_col.append(i)


merge_df = pd.merge(internal_df,external_df,on='PROSPECTID',how='inner')
num_cols = [i for i in merge_df.columns if merge_df[i].dtype == 'int64']
cat_col = [i for i in merge_df.columns if merge_df[i].dtype == 'object']
contineous_col = [i for i in merge_df.columns if merge_df[i].dtype == 'float64']

not_cat_var = set(merge_df.columns) - set(cat_col)

# Feature Selction 
# applying chi square test for categorical variable importance on target variable 

chi_score = []
p_score = []
dof_score = []
expected_score = []

for i in cat_col[:-1]:
    contingency_table = pd.crosstab(merge_df[i], merge_df['Approved_Flag'])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    chi_score.append(chi2)
    p_score.append(p_val)
    dof_score.append(dof)
    expected_score.append(expected)


import pandas as pd
# p_values = pd.Series(chi_score[1],index = merge_df.columns)
# p_score = pd.DataFrame(p_score)
# p_score 
# p_score.sort_values(ascending = False , inplace = True)
# p_score.plot.bar()

p_values_df = pd.DataFrame({
    'Variable': cat_col[:-1],
    'p_value': p_score
})

# Sort the DataFrame by p-values
p_values_df_sorted = p_values_df.sort_values(by='p_value', ascending=False)

# Plot the sorted p-values as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(p_values_df_sorted['Variable'], p_values_df_sorted['p_value'])
plt.xlabel('Categorical Variables')
plt.ylabel('p-value')
plt.title('Chi-square p-values for Categorical Variables')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# for numerical variables 
# 1. Using Anova test 
from sklearn.feature_selection import f_classif
x1 = merge_df.drop('Approved_Flag',axis= 1)
x1 = merge_df.drop(columns=cat_col,axis=1)
y1 = merge_df['Approved_Flag']

from scipy.stats import f_oneway
columns_need_kept = []

for i in not_cat_var :
    
    a = list(merge_df[i])
    b = list(merge_df['Approved_Flag'])
    p1_grp = [val for val, grp in zip(a,b) if grp == 'P1' ]
    p2_grp = [val for val, grp in zip(a,b) if grp == 'P2' ]
    p3_grp = [val for val, grp in zip(a,b) if grp == 'P3' ]
    p4_grp = [val for val, grp in zip(a,b) if grp == 'P4' ]
    
    f_state, p_val = f_oneway(p1_grp,p2_grp,p3_grp,p4_grp)
    if p_val <= 0.05 :
        columns_need_kept.append(i)
        
           
columns_need_kept = list(columns_need_kept)


# Basic EDA
discription = merge_df.describe()
features = columns_need_kept + cat_col 
df = merge_df[features]
final_cols = df.columns

# BASIC EDA ( EXPLORATORY DATA ANALYSIS)
feature_data = df.copy()

feature_data['MARITALSTATUS'].unique()
feature_data['EDUCATION'].unique()
feature_data['GENDER'].unique()
feature_data['last_prod_enq2'].unique()
feature_data['first_prod_enq2'].unique()
# target_var.unique()
sorted_col = sorted(feature_data.columns)

# BASIC EDA 
education_cat = feature_data['EDUCATION'].value_counts()
marrital_status_cat = feature_data['MARITALSTATUS'].value_counts()
gender_cat = feature_data['GENDER'].value_counts()
last_prod_enq2_cat = feature_data['last_prod_enq2'].value_counts()
first_prod_enq2_cat = feature_data['first_prod_enq2'].value_counts()
# count plot for categorical variable 

count_list = [education_cat,marrital_status_cat,gender_cat,last_prod_enq2_cat,first_prod_enq2_cat]

plt.figure(figsize=(8,6)) # PLOT 1 
sns.barplot(x= education_cat.index,y= education_cat.values, palette='Blues_d')
plt.title('EDUCATION WISE DISTRIBUTION')
plt.xlabel('EDUCATION CATEGORY')
plt.ylabel('COUNT OF PROSPECT ID')
plt.xticks(rotation=30, ha='right')

# PLOT 2 
sns.barplot(x= first_prod_enq2_cat.index,y= first_prod_enq2_cat.values, palette='Blues_d')
plt.title('first_prod_enq2_cat WISE DISTRIBUTION')
plt.xlabel('first_prod_enq2_cat CATEGORY')
plt.ylabel('CONUT OF PROSPECT ID')
plt.xticks(rotation=30, ha='right')

monthly_income = feature_data['NETMONTHLYINCOME']
credit_score = feature_data['Credit_Score']

# variation of montly income 
# PLOT 3 
# sns.histplot(monthly_income,bins=50,kde=True,color='green')
# plt.title('Monthly Income Variation')
# plt.xlabel('Monthly Income')
# plt.ylabel('Density')

# PLOT 4 
# sns.barplot(x= marrital_status_cat.index,y= marrital_status_cat.values, palette='Blues_d')
# plt.title('MARITALSTATUS WISE DISTRIBUTION')
# plt.xlabel('MARITALSTATUS CATEGORY')
# plt.ylabel('CONUT OF PROSPECT ID')
# plt.xticks(rotation=30, ha='right')

auto_tl_cat = feature_data['Auto_TL'].value_counts()
cc_tl_cat = feature_data['CC_TL'].value_counts()
consumer_tl_cat = feature_data['Consumer_TL'].value_counts()
gold_tl_cat = feature_data['Gold_TL'].value_counts()
home_tl_cat = feature_data['Home_TL'].value_counts()
pl_tl_cat = feature_data['PL_TL'].value_counts()
secured_tl_cat = feature_data['Secured_TL'].value_counts()
unsecured_tl_cat = feature_data['Unsecured_TL'].value_counts()
other_tl_cat = feature_data['Other_TL'].value_counts()

# PLOT 5 
# sns.barplot(x= auto_tl_cat.index,y= auto_tl_cat.values, palette='Blues_d')
# plt.title('auto_tl WISE DISTRIBUTION')
# plt.xlabel('Auto TL CATEGORY')
# plt.ylabel('COUNT OF PROSPECT ID')
# plt.xticks(rotation=30, ha='right')

# derived attribute total_no_of_account 

total_in_auto_tl = sum(feature_data['Auto_TL'])

ALL_TL_CAT = ['Auto_TL', 'CC_TL', 'Consumer_TL', 'Gold_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL' ]
ALL_TL_CAT_LABEL = ['Automobile Accounts', 'Credit Card Accounts', 'Consumer goods Accounts','Gold loan Accounts','Housing loan Accounts','Personal Loan Accounts','Secured Accounts','Unsecured Accounts','Other Accounts']


sum_lst = []
for i in ALL_TL_CAT:
    sum_lst.append(sum(feature_data[i]))
    
# PLOT 6 
# sns.barplot(x=ALL_TL_CAT, y = sum_lst)
# plt.xlabel('ALL types of Accounts ')
# plt.ylabel('Count of Prospect ID (Not Unique)')
# plt.xticks(rotation = 30, ha = 'right')

last_6m = [col for col in feature_data.columns if '6m' in col or '6M' in col]
last_6m_without_percent = [col for col in last_6m if 'pct' not in col]
last_6m_without_percent_label = ['Credit card enquiries','Number of substandard payments', 'Total accounts closed', 'Enquiries'
                                 , 'Number of doubtful payments', 'Total accounts opened ','Number of standard Payments', 'Personal Loan enquiries','Number of times delinquent']

last_6m_list_wp = []
for i in last_6m_without_percent :
    last_6m_list_wp.append(sum(feature_data[i]))

# PLOT 7 
# sns.lineplot(x=last_6m_without_percent, y= last_6m_list_wp)
# plt.xlabel('last 6m data')
# plt.ylabel('sum of last 6m data')
# plt.xticks(rotation=30, ha ='right')

# PLOT 8
# sns.barplot(x=last_6m_without_percent, y= last_6m_list_wp)
# plt.xlabel('last 6m data')
# plt.ylabel('sum of last 6m data')
# plt.xticks(rotation=30, ha ='right')

# last 12 month  related data 

last_12m = [col for col in feature_data.columns if '12m' in col or '12M' in col]
last_12m_without_percent = [col for col in last_12m if 'pct' not in col]

last_12m_list_wp = []
for i in last_12m_without_percent :
    last_12m_list_wp.append(sum(feature_data[i]))
    
last_12m_without_percent_label = ['Number of times of missed payments  between last 6 and 12 months', 'Credit card enquiries', 'Enquiries', 'Number of standard Payments',
                                  'Total accounts closed', 'Total accounts opened', 'Personal Loan enquiries',
                                  'Number of substandard payments','Number of doubtful payments', 'Number of times delinquent ']

pct_data = [col for col in feature_data.columns if 'pct' in col]
npct_cols = [ col for col in feature_data.columns if col not in pct_data]

credit_card_data_col = [col for col in feature_data.columns if 'cc' in col or 'CC' in col and 'pct' not in col]
personal_loan_data_col = [col for col in feature_data.columns if 'pl' in col or 'PL' in col and 'pct' not in col]
housing_loan_data_col = ['HL_Flag']
loss_accounts_data_col = [col for col in feature_data.columns if 'lss' in col or 'LSS' in col]
doubtful_payments_col = [col for col in feature_data.columns if 'dbt' in col or 'DBT' in col]
substandard_payments = [col for col in feature_data.columns if 'sub' in col or 'SUB' in col]
delinquent_data = [col for col in feature_data.columns if 'deliq' in col or 'DELIQ' in col]
total_account_data = [col for col in feature_data.columns if '_TL_' in col]

# credit card data plot
credit_card_data_sum = []
for i in credit_card_data_col:
    credit_card_data_sum.append(sum(feature_data[i]))

personal_loan_data_sum = []
for i in personal_loan_data_col:
    personal_loan_data_sum.append(sum(feature_data[i]))


delinquent_data_sum = []
for i in delinquent_data:
    if 'level_of' not in i:
        delinquent_data_sum.append(sum(feature_data[i]))
    

total_account_data_sum = []
for i in total_account_data:
    total_account_data_sum.append(sum(feature_data[i]))

feature_data = feature_data[feature_data['NETMONTHLYINCOME'] < 2.500000e+06]

# COMPLETED EDA 
# MODEL BUILDING USING FEATURE SELECTION 
new_desc = feature_data.describe()

# Need to do encoding for categorical features 
encoded_df  = feature_data.copy()
# for Gender male --> 1 , female --> 0
# for marital status married --> 1 , single --> 0

from sklearn import preprocessing
gender_label = preprocessing.LabelEncoder()

encoded_df['GENDER'] = gender_label.fit_transform(encoded_df['GENDER'])
marital_status_mapping = {'Single':0, 'Married' : 1}
encoded_df['MARITALSTATUS'] = encoded_df['MARITALSTATUS'].map(marital_status_mapping)

# print(encoded_df.head(3))

encoded_df = pd.get_dummies(encoded_df, columns=['EDUCATION'])

# print(encoded_df.shape)

# print(df['last_prod_enq2'].value_counts())
encoded_df = pd.get_dummies(encoded_df, columns=['last_prod_enq2'])
encoded_df = pd.get_dummies(encoded_df, columns=['first_prod_enq2'])


from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
feature_data_2 = encoded_df.copy()

x = encoded_df.drop(['Approved_Flag','PROSPECTID'],axis=1)
y = encoded_df['Approved_Flag']
ont_set = set()
ont_set.add('Approved_Flag')
ont_set.add('PROSPECTID')
df_colmuns = set(encoded_df.columns) - ont_set
x_copy1 = x.copy()
y_copy1 = y.copy()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

min_max_scaler = StandardScaler()

x = min_max_scaler.fit_transform(x)
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)


x_train_mi = x_train.copy()
x_test_mi = x_test.copy()
y_train_mi = y_train_encoded.copy()
y_test_mi = y_test_encoded.copy()

x_train_dt = x_train.copy()
x_test_dt = x_test.copy()
y_train_dt = y_train_encoded.copy()
y_test_dt = y_test_encoded.copy()

x_train_rf = x_train.copy()
y_train_rf = y_train_encoded.copy()
x_test_rf = x_test.copy()
y_test_rf = y_test_encoded.copy()

x_train_xgb = x_train.copy()
x_test_xgb = x_test.copy()
y_train_xgb = y_train_encoded.copy()
y_test_xgb = y_test_encoded.copy()

min_max_scaled_data_x = min_max_scaler.fit_transform(x)
min_max_scaled_data_x_train = x_train.copy()
min_max_scaled_data_x_test = x_test.copy()

# min_max_scaled_data_y_train = min_max_scaler.fit_transform(y_train)


# want to know the best featues 
# Method 1 : mututal information 
from sklearn.feature_selection import mutual_info_classif, SelectKBest
best_features_using_mutual = SelectKBest(score_func= mutual_info_classif,k=35)
fit = best_features_using_mutual.fit(min_max_scaled_data_x,y)
# Get the scores for each feature
columns_name = list(df_colmuns)
feature_scores = pd.DataFrame(fit.scores_ , index= columns_name, columns=['Score'])
selected_features = feature_scores.nlargest(35, 'Score')  # Change '10' to the number of features you want to select

plt.figure(figsize=(8,9))
sns.barplot(x=selected_features['Score'], y=selected_features.index, palette='viridis')
plt.title('Top 35 Features Based on mutual information Score')
plt.xlabel('Score')
plt.ylabel('Features')
plt.show()

# USING DECISION TREE CLASSIFIER 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

decision_model = DecisionTreeClassifier(random_state=42)
decision_model.fit(min_max_scaled_data_x_train, y_train)
feature_importance = decision_model.feature_importances_
feature_scores = pd.DataFrame({'Featuer': columns_name , 'Importance': feature_importance})
feature_scores = feature_scores.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(6,10))
sns.barplot(x='Importance', y='Featuer', data=feature_scores.head(35))
plt.title('Feature Importances of top 35 features using Decision Tree Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
# Conclusion : only feature 'Credit Score' is having more weightage than other features


# using random forest classifier 
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators=30,criterion='entropy')
random_forest_classifier.fit(x_train_rf,y_train_rf)
y_pred_rf = random_forest_classifier.predict(x_test_rf)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
random_forest_cm = confusion_matrix(y_test_rf, y_pred_rf)

disp = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(y_test_rf, y_pred_rf), display_labels=['P1','P2','P3', 'P4'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix Using Random Forest Classifier")
plt.show()


accuracy = accuracy_score(y_test_rf, y_pred_rf)
print(f'Accuracy using random forest classifier: {accuracy}')
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_rf, y_pred_rf)


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

plt.figure(figsize=(8, 6))
sns.heatmap(random_forest_cm , annot=True, fmt='d', cmap='Blues', xticklabels=['P0', 'P1','P2','P3'], yticklabels=['P0', 'P1','P2','P3'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Using XGBoost 
import xgboost as xgb 
# need to label the target variable 
from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train_xgb)
# y_test_encoded = label_encoder.fit_transform(y_test_xgb)
# x_train3, x_test3, y_train3, y_test3 = train_test_split(x_copy1 , y_encoded, test_size=0.2, random_state=42)
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class = 4)
xgb_classifier.fit(x_train_xgb, y_train_encoded)
y_pred_xgb = xgb_classifier.predict(x_test_xgb)

# Decode the predicted labels back to original format (if needed)
y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

# Accuracy check using XG Boost 
xgb_accuracy = accuracy_score(y_test_xgb, y_pred_xgb)
print(f'Accuracy using XG Boost: {xgb_accuracy:.2f}')

precision_xgb, recall_xgb, f1_score_xgb, _xgb = precision_recall_fscore_support(y_test_xgb , y_pred_xgb,labels= ['0','1','2','3'])

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision_xgb[i]}")
    print(f"Recall: {recall_xgb[i]}")
    print(f"F1 Score: {f1_score_xgb[i]}")
    print()
    
conf_matrix_xgb = confusion_matrix( y_test_xgb, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb  , annot=True, fmt='d', cmap='Blues', xticklabels=['P0', 'P1','P2','P3'], yticklabels=['P0', 'P1','P2','P3'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Using XG-Boost')
plt.show()

# USING DECISION TREE 
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train_dt, y_train_encoded)
y_pred_dt = dt_model.predict(x_test_dt)

# Decode the predicted labels back to original format (if needed)
y_pred_dt_decoded = label_encoder.inverse_transform(y_pred_dt)


# Accuracy using decision tree 
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt_decoded)
print(f"Accuracy using Decision Tree: {accuracy_dt:.2f}")
precision_dt, recall_dt, f1_score_dt, _dt = precision_recall_fscore_support(y_test_dt, y_pred_dt)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision_dt[i]}")
    print(f"Recall: {recall_dt[i]}")
    print(f"F1 Score: {f1_score_dt[i]}")
   
    
conf_matrix_dt = confusion_matrix(y_test_dt, y_pred_dt)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_dt  , annot=True, fmt='d', cmap='Blues', xticklabels=['P0', 'P1','P2','P3'], yticklabels=['P0', 'P1','P2','P3'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Using Decision Tree')
plt.show()


# classificatino report 
from sklearn.metrics import classification_report
xgb_classification_report = classification_report(y_test_xgb, y_pred_xgb, target_names=['P1','P2', 'P3', 'P4']) #
classification_report_rf = classification_report(y_test_rf,y_pred_rf)
dt_classification_report = classification_report(y_test_dt, y_pred_dt)


# # ROC-AUC Curve for multi class classifier 
# # 2 types : One-vs-Rest multiclass ROC  (OVR ROC)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
# for the data of Random Forest 

# ROC CURVE FOR RANDOM FOREST CLASSIFIER 
y_prob_rf = random_forest_classifier.predict_proba(x_test_rf) # # Predict probabilities using the existing model
classes = ['0', '1', '2', '3'] 
y_test_rf = y_test_rf.astype(str)
y_test_binarized = label_binarize(y_test_rf, classes=classes) # Binarize the output for each class separately (One-vs-Rest)
    
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob_rf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(6,4))
colors = ['blue', 'green', 'red', 'purple']
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
              label=f'ROC curve of class P{i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


# ROC CURVE FOR XG BOOST ALGORITHM 
y_prob_xgb = random_forest_classifier.predict_proba(x_test_xgb ) # # Predict probabilities using the existing model


classes = ['0', '1', '2', '3']
y_test_xgb = y_test_rf.astype(str)
y_test_binarized_xgb = label_binarize(y_test_xgb, classes=classes) # Binarize the output for each class separately (One-vs-Rest)
    
# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized_xgb[:, i], y_prob_xgb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(6,4))
colors = ['blue', 'green', 'red', 'purple']
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
              label=f'ROC curve of class P{i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XG Boost Classifier')
plt.legend(loc="lower right")
plt.show()



# ROC CURVE FOR DECISION TREE 
y_prob_dt = random_forest_classifier.predict_proba(x_test_dt ) # # Predict probabilities using the existing model
# Binarize the output for each class separately (One-vs-Rest)
classes = ['0', '1', '2', '3']
y_test_dt = y_test_rf.astype(str)
y_test_binarized_dt = label_binarize(y_test_dt, classes=classes)
    
# Plot ROC curves
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized_dt[:, i], y_prob_dt[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(6,4))
colors = ['blue', 'green', 'red', 'purple']
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
              label=f'ROC curve of class P{i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree Classifier')
plt.legend(loc="lower right")
plt.show()

false_prediction = []
for i in range(len(y_test_rf)):
    
    if y_test_rf[i] != str(y_pred_rf[i]):
        false_prediction.append({i : [int(y_test_rf[i]),y_pred_rf[i]]})
        
        
# unseen data 

unseen_df = pd.read_excel("D:\\My books\\cibil_dataset_kaggle\\Unseen_Dataset.xlsx")









# Dash 

import dash 
from dash import dash_table, dcc, html, Input, Output

import plotly.graph_objects as go 
import plotly.express as px
from plotly.subplots import make_subplots


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the page
layout_page1 = html.Div(style={'backgroundColor': '#000000', 'height': '100vh'},children =  [
    html.H1("EDA of CIBIL DATA",style={'textAlign': 'center', 'color': '#FFFFFF'}),
    dash_table.DataTable(
        id='id-data-table', 
        columns=[{"name": i, "id": i} for i in merge_df.columns], 
        data= merge_df.to_dict('records'), 
        style_table={'overflowY': 'auto', 'maxHeight': '400px', 'overflowX': 'auto'},
        filter_action='native',
        fixed_rows={'headers': True},
        style_cell={'textAlign': 'center', 'minWidth': '50px', 'width': '100px', 'maxWidth': '200px'}
    ),
    
    dcc.Graph(id='plot-1',style={'backgroundColor': '#000000'}),
    dcc.Graph(id='plot-2'),
    dcc.Graph(id='plot-3'),
    dcc.Graph(id = 'plot-4'),
    dcc.Graph(id = 'plot-5'),
    dcc.Graph(id = 'plot-6'),
    dcc.Graph(id = 'plot-7'),
    dcc.Graph(id = 'plot-8'),
    dcc.Graph(id = 'plot-9'),
    dcc.Graph(id = 'plot-10'),
    dcc.Graph(id = 'plot-11'),
    dcc.Graph(id = 'plot-12'),
    dcc.Graph(id = 'plot-13'),
    dcc.Graph(id = 'plot-14'),
    dcc.Graph(id = 'plot-15'),
    dcc.Graph(id = 'plot-16'),
    dcc.Graph(id = 'plot-17'),
    dcc.Graph(id = 'plot-18'),
    dcc.Graph(id = 'plot-19'),
    dcc.Graph(id = 'plot-20'),
    dcc.Graph(id = 'plot-21')
    
    # dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
])

# Define the callback for updating the plots
@app.callback(
    [ Output('plot-1', 'figure'),
      Output('plot-2', 'figure'),
      Output('plot-3', 'figure'),
      Output('plot-4', 'figure'),
      Output('plot-5', 'figure'),
      Output('plot-6', 'figure'),
      Output('plot-7', 'figure'),
      Output('plot-8', 'figure'),
      Output('plot-9', 'figure'),
      Output('plot-10', 'figure'),
      Output('plot-11', 'figure'),
      Output('plot-12', 'figure'),
      Output('plot-13', 'figure'),
      Output('plot-14', 'figure'),
      Output('plot-15', 'figure'),
      Output('plot-16', 'figure'),
      Output('plot-17', 'figure'),
      Output('plot-18', 'figure'),
      Output('plot-19', 'figure'),
      Output('plot-20', 'figure'),
      Output('plot-21', 'figure')],
    
    [Input('url', 'pathname')]
    # [Input('category-dropdown', 'value')
    # ]
)
def plotting(pathname):
    plot1 = px.bar(x=education_cat.index, y=education_cat.values, title='EDUCATION WISE DISTRIBUTION',color=education_cat.index)
    plot1.update_layout(
        paper_bgcolor='black', 
        plot_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='grey'),
        yaxis=dict(color='white', gridcolor='grey')
    )
    plot1.update_xaxes(title = 'Education level')
    plot1.update_yaxes(title = 'Count of prospect id')
    
    plot2 = px.bar(x=['others', 'consumer loan accounts','personal loan accounts','Automobile Accounts','Credit Card Accounts','Housing loan Accounts'], y=first_prod_enq2_cat.values ,color = first_prod_enq2_cat.index, title=' Distribution according to First product enquired for')
    # first_prod_enq2_cat.values
    # y=['others', 'consumer loan accounts','personal loan accounts','Automobile Accounts','Credit Card Accounts','Housing loan Accounts']
    # others consumer goods accounts, , personal loan accounts, Automobile accounts, Credit Card accounts, Housing loan accounts 
    
    plot2.update_layout(
        paper_bgcolor='black', 
        plot_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='grey'),
        yaxis=dict(color='white', gridcolor='grey')
    )
    plot2.update_xaxes(title = 'first_prod_enq2_cat type')
    
    monthly_income = feature_data['NETMONTHLYINCOME']
    credit_score = feature_data['Credit_Score']
    plot3 = px.histogram(monthly_income, nbins=50, title='Monthly Income Variation')
    plot3.update_layout(
        paper_bgcolor='black', 
        plot_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='grey'),
        yaxis=dict(color='white', gridcolor='grey')
    )
    plot3.update_xaxes(title= 'Net Monthly Income')
    plot3.update_yaxes(title = 'Count of Prospect ID')

    plot4 = px.bar(x= marrital_status_cat.index,y= marrital_status_cat.values, color= marrital_status_cat.index , title='MARITALSTATUS WISE DISTRIBUTION')
    plot4.update_layout(
        paper_bgcolor='black', 
        plot_bgcolor='black',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='grey'),
        yaxis=dict(color='white', gridcolor='grey')
    )
    plot4.update_xaxes(title= 'Marritcal status')
    plot4.update_yaxes(title = 'Count of prospect id')
    
    
    plot5 = px.bar(x= auto_tl_cat.index,y= auto_tl_cat.values, color= auto_tl_cat.index , title='Distribution according to Count of Automobile accounts')
    plot5.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    plot5.update_xaxes(title = 'Count of Autombiles Accouts')
    plot5.update_yaxes(title = 'Count of prospect id')
    
    # PLOT 6 
    plot6 = px.bar(x=ALL_TL_CAT_LABEL, y = sum_lst, title='All types of accounts', color= ALL_TL_CAT_LABEL)
    plot6.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    plot6.update_xaxes(title = 'All types of accounts')
    plot6.update_yaxes(title = 'total sum of accounts')
    
    # Plot 7 
    plot7 = px.line(x=last_6m_without_percent_label, y= last_6m_list_wp)
    plot7.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    plot7.update_xaxes(title = 'last 6 month data')
    plot7.update_yaxes(title = 'Count of Prospect ID')
    
    
    # PLOT 8
    plot8 = px.bar(x=last_6m_without_percent_label, y= last_6m_list_wp, color= last_6m_without_percent_label)
    plot8.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot8.update_xaxes(title = 'last 6m data')
    plot8.update_yaxes(title = 'Count of Prospect ID')
    
    # PLOT 9
    plot9 = px.bar(x=last_12m_without_percent_label, y = last_12m_list_wp) #, color= last_12m_without_percent_label
    plot9.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    plot9.update_xaxes(title = 'last 12 month data')
    
    
    plot10 = px.bar(x= gender_cat.index,y= gender_cat.values, title= 'GENDER WISE DISTRIBUTION', color= gender_cat.index)
    plot10.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot10.update_xaxes(title = 'Gender')
   
    
    # PLOT 11 
    plot11 = px.scatter(x=feature_data['Credit_Score'], y=feature_data['Time_With_Curr_Empr'], title='Credit score v/s tenure', color= feature_data['Approved_Flag'])
    plot11.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot11.update_xaxes(title = 'Credit Score')
    plot11.update_yaxes(title = 'Time with curr Employer')
    
    plot12 = px.scatter(x=feature_data['Credit_Score'], y=feature_data['NETMONTHLYINCOME'], title= 'Credit Score with net monthly income', color= feature_data['Approved_Flag'])
    plot12.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot12.update_xaxes(title = 'Crdit Score')
    plot12.update_yaxes(title = 'Net Monthly Income')
    
    # PLOT 13
    plot13 = px.box(x=feature_data['NETMONTHLYINCOME'])
    plot13.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot13.update_xaxes(title = 'net monthly income')
    
    # PLOT 14
    plot14 = px.scatter(x=feature_data['num_std'], y=feature_data['NETMONTHLYINCOME'], color= feature_data['Approved_Flag'])
    plot14.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot14.update_xaxes(title = 'Number of standard Payments')
    plot14.update_yaxes(title = 'Net monthly income')
    
    # PLOT 15
    plot15 = px.pie(credit_card_data_sum, names=credit_card_data_col, title= 'Pie Plot for credit data')
    plot15.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot15.update_xaxes(title = 'Credit card data')
    
    plot16 = px.pie(personal_loan_data_sum, names=personal_loan_data_col, title='Pie plot for personal loan')
    plot16.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot16.update_xaxes(title = 'personal loan data')
    
    # PLOT 17
    plot17 = px.pie(delinquent_data_sum, names= [i for i in delinquent_data if 'level_of' not in i], title='Pie plot for Delinquent data')
    plot17.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot17.update_xaxes(title = 'delinquent data')
    # PLOT 18
    plot18 = px.pie(total_account_data_sum, names= total_account_data, title= 'Pie plot for Total account Data')
    plot18.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot18.update_xaxes(title = 'total account data')
    # PLOT 19
    plot19 = px.scatter(x=feature_data['Credit_Score'], y=feature_data['Time_With_Curr_Empr'], color= feature_data['GENDER'])
    plot19.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot19.update_xaxes(title = 'Credit Score')
    plot19.update_yaxes(title = 'Time with current Employer')

    # PLOT 20
    plot20 = px.bar(x= last_prod_enq2_cat.index,y= last_prod_enq2_cat.values, title= 'Distribution according to Latest product enquired for', color = last_prod_enq2_cat.index)
    plot20.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )
    
    plot20.update_xaxes(title = 'Latest product enquired for')
    
    # PLOT 21
    # plt.figure(figsize=(8,10))
    plot21 = px.histogram(feature_data['NETMONTHLYINCOME'],title='Monthly income variation')
    plot21.update_layout(
       paper_bgcolor='black', 
       plot_bgcolor='black',
       font=dict(color='white'),
       title_font=dict(color='white'),
       xaxis=dict(color='white', gridcolor='grey'),
       yaxis=dict(color='white', gridcolor='grey')
   )

    plot21.update_xaxes(title = 'Net monthly income')
    
    return plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12, plot13, plot14, plot15, plot16, plot17, plot18, plot19, plot20, plot21        

# Define the callback for page navigation
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/cibil_data':
        return layout_page1
    else:
        return '404 - Page not found'

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Run the server
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050)
