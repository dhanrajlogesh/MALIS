#!/usr/bin/env python
# coding: utf-8

# In[58]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


# In[59]:


df=pd.read_csv(r"/Users/jeroldmain/Downloads/heart_2020_cleaned.csv")
df.shape


# In[60]:


df.head()


# In[61]:


df.duplicated().sum()


# In[62]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[63]:


df.isna().sum()


# In[64]:


df.info()


# In[65]:


df.nunique()


# In[66]:


# Label Encoding
object_feature = df.dtypes[df.dtypes == 'O'].index.values
L = LabelEncoder()

for i in object_feature:
    df[i] = L.fit_transform(df[i])
df


# In[67]:


df.info()


# In[68]:


# Dropping race column in the dataset
df = df.drop('Race', axis=1)
df.info()


# In[69]:


df.shape


# In[70]:


# Heatmap
corr = df.corr()
top_feature = corr.index
plt.subplots(figsize=(12, 8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


# In[71]:


# Class Imbalance
sns.countplot(x='HeartDisease',data = df)


# In[72]:


# UnderSampling
# count class
count_class_0, count_class_1 = df.HeartDisease.value_counts()
class_0 = df[df['HeartDisease'] == 0]
class_1 = df[df['HeartDisease'] == 1]


# In[73]:


count_class_0, count_class_1


# In[74]:


class_0.shape


# In[75]:


class_1.shape


# In[76]:


data_us_0 = class_0.sample(count_class_1)
us_data = pd.concat([data_us_0, class_1], axis=0)
us_data.shape
print(us_data.HeartDisease.value_counts())


# In[77]:


# Making x,y undersampling
x = us_data.drop('HeartDisease',axis='columns')
y = us_data['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15, stratify=y)
y_train.value_counts()


# In[78]:


# Implementation of Logistic Regression
logistic =LogisticRegression() 
logistic.fit(x_train,y_train)
prediction = logistic.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(logistic, x_test, y_test, cmap='Blues',values_format='.3g')


# In[79]:


# Implementation of KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
prediction = knn.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(knn, x_test, y_test, cmap='Blues',values_format='.3g')


# In[80]:


# Implementation of Decision Tree
dt = tree.DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)
prediction = dt.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(dt, x_test, y_test, cmap='Blues',values_format='.3g')


# In[81]:


# Implementation of Random Forest
rf =RandomForestClassifier() 
rf.fit(x_train,y_train)
prediction = rf.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(rf, x_test, y_test, cmap='Blues',values_format='.3g')


# In[82]:


# ROC Curve Undersampling
logistic_probs = logistic.predict_proba(x_test)
knn_probs = knn.predict_proba(x_test)
dt_probs = dt.predict_proba(x_test)
rf_probs = rf.predict_proba(x_test)

logistic_probs = logistic_probs[:, 1]
knn_probs = knn_probs[:, 1]
dt_probs = dt_probs[:, 1]
rf_probs = rf_probs[:, 1]
logistic_auc = roc_auc_score(y_test, logistic_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
print('Logistic Regression: AUROC = %.3f' % (logistic_auc))
print('KNN: AUROC = %.3f' % (knn_auc))
print('Decission Tree: AUROC = %.3f' % (dt_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))


# In[83]:


logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic regression (AUROC = %0.3f)' % logistic_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN (AUROC = %0.3f)' % knn_auc)
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decission Tree (AUROC = %0.3f)' % dt_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)

plt.title('ROC curve for Under Sampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend() 
plt.show()


# In[84]:


# Oversampling

data_os_1 = class_1.sample(count_class_0, replace=True)
os_data = pd.concat([data_os_1, class_0], axis=0)
os_data.shape
print(os_data.HeartDisease.value_counts())


# In[85]:


# making x,y oversampling
x = os_data.drop('HeartDisease',axis='columns')
y = os_data['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15, stratify=y)
y_test.value_counts()


# In[86]:


# Implementing Logistic Regression
logistic =LogisticRegression() 
logistic.fit(x_train,y_train)
prediction = logistic.predict((x_test))
print('Mean Square Error testing model ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(logistic, x_test, y_test, cmap='Blues',values_format='.3g')


# In[87]:


# Implementation of KNN 
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
prediction = knn.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(knn, x_test, y_test, cmap='Blues',values_format='.3g')


# In[88]:


# Implementing Decision Tree
dt = tree.DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)
prediction = dt.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(dt, x_test, y_test, cmap='Blues',values_format='.3g')


# In[89]:


# Implementation of Random Forest
rf =RandomForestClassifier() 
rf.fit(x_train,y_train)
prediction = rf.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(rf, x_test, y_test, cmap='Blues',values_format='.3g')


# In[90]:


#ROC Curve Over Sampling
logistic_probs = logistic.predict_proba(x_test)
knn_probs = knn.predict_proba(x_test)
dt_probs = dt.predict_proba(x_test)
rf_probs = rf.predict_proba(x_test)

logistic_probs = logistic_probs[:, 1]
knn_probs = knn_probs[:, 1]
dt_probs = dt_probs[:, 1]
rf_probs = rf_probs[:, 1]

logistic_auc = roc_auc_score(y_test, logistic_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
print('Logistic Regression: AUROC = %.3f' % (logistic_auc))
print('KNN: AUROC = %.3f' % (knn_auc))
print('Decission Tree: AUROC = %.3f' % (dt_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))


# In[91]:


logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic regression (AUROC = %0.3f)' % logistic_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN (AUROC = %0.3f)' % knn_auc)
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decission Tree (AUROC = %0.3f)' % dt_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.title('ROC curve for Over Sampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend() 
plt.show()


# In[92]:


# SMOTE
x = us_data.drop('HeartDisease',axis='columns')
y = us_data['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15, stratify=y)
y_train.value_counts()
smote = SMOTE(random_state=42)
x_sm, y_sm = smote.fit_resample(x,y)


# In[93]:


# Implementation of Logistic Regression
logistic =LogisticRegression() 
logistic.fit(x_sm,y_sm)
prediction = logistic.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(logistic, x_test, y_test, cmap='Blues',values_format='.3g')


# In[94]:


# Implementation of KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_sm, y_sm)
prediction = knn.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(knn, x_test, y_test, cmap='Blues',values_format='.3g')


# In[95]:


# Implementation of Decision Tree
dt = tree.DecisionTreeClassifier(random_state=0)
m= dt.fit(x_sm, y_sm)
prediction = dt.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(dt, x_test, y_test, cmap='Blues',values_format='.3g')


# In[96]:


# Implementation of Random Forest
rf =RandomForestClassifier() 
rf.fit(x_sm,y_sm)
prediction = rf.predict((x_test))
print('Mean Square Error testing model  ', metrics.mean_squared_error(y_test, prediction))
print("Classification Report: \n", classification_report(y_test, prediction))
print('confusion matrix: \n',confusion_matrix(y_test, prediction))
show = plot_confusion_matrix(rf, x_test, y_test, cmap='Blues',values_format='.3g')


# In[98]:


# AUROC Curve for SMOTE
logistic_probs = logistic.predict_proba(x_test)
knn_probs = knn.predict_proba(x_test)
dt_probs = dt.predict_proba(x_test)
rf_probs = rf.predict_proba(x_test)

logistic_probs = logistic_probs[:, 1]
knn_probs = knn_probs[:, 1]
dt_probs = dt_probs[:, 1]
rf_probs = rf_probs[:, 1]

logistic_auc = roc_auc_score(y_test, logistic_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
print('Logistic Regression: AUROC = %.3f' % (logistic_auc))
print('KNN: AUROC = %.3f' % (knn_auc))
print('Decission Tree: AUROC = %.3f' % (dt_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))


# In[99]:


logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic regression (AUROC = %0.3f)' % logistic_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN (AUROC = %0.3f)' % knn_auc)
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decission Tree (AUROC = %0.3f)' % dt_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.title('ROC curve for SMOTE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend() 
plt.show()

