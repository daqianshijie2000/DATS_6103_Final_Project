import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#%%-------------------------------------------------------------------------------------
#read two datasets
df_train = pd.read_csv('/Users/daqian.dang/Desktop/DATS 6013/Final Project/train.csv')
df_test = pd.read_csv('/Users/daqian.dang/Desktop/DATS 6013/Final Project/test.csv')
#Merge two datasets
df = pd.concat([df_train,df_test])
#%%-------------------------------------------------------------------------------------
# print the dataset observations
print('First ten rows of the dataset:\n')
print(df.head(10))

# print the dataset shape
print("Dataset number of rows: ", df.shape[0])
print("Dataset number of columns: ", df.shape[1])

# printing the struture of the dataset
print("The dataset info:\n ")
print(df.info())

# print the summary statistics of the dataset
print(df.describe(include='all'))

# print the dataset features
print(df.columns)

#%%-------------------------------------------------------------------------------------
# data preprocessing
# specify the numeric variables
numeric_var = ['id','Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']
# convert the specified variables into numeric
df[numeric_var] = df[numeric_var].apply(pd.to_numeric, errors ='coerce')

# find and count the number of missing values
print(df.isnull().sum())
# replace empty spaces to NaN
df.replace('', np.NaN, inplace=True)
# use median imputation to convert the missing values
median_imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
df[numeric_var] = median_imputer.fit_transform(df[numeric_var])

#%%-------------------------------------------------------------------------------------
# Basic data analysis

# Age mean for gender
print('mean age per gender:\n', df.groupby('Gender').Age.mean())
# Total number of males and females
print('Total number of males and females:\n', df.groupby('Gender').Gender.count())
# Total number of males and females per satisfaction class
print('Total number of males and females per satisfaction class:\n', df.groupby(['Gender','satisfaction']).Gender.count())
#Total number of males and females for the combination of gender and class
print('Total number of males and females per combination of gender and class:\n', df.groupby(['Gender','Class']).Gender.count())
#Total number of males and females for the combination of gender and custormer type
print('Total number of males and females per combination of gender and customer type:\n', df.groupby(['Gender','Customer Type']).Gender.count())
#%%-------------------------------------------------------------------------------------
# Numeric Variables visualization
df['Age'].hist(bins=30, grid=False, color='red')
plt.title('Age Per Frequency')
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

df['Flight Distance'].hist(bins=50, grid=False, color='red')
plt.title('Flight Distance Per Frequency')
plt.xlabel('Flight Distance', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

df['Departure Delay in Minutes'].hist(bins=150, grid=False, color='red')
plt.title('Departure Delay in Minutes Per Frequency')
plt.xlabel('Departure Delay in Minutes', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

df['Arrival Delay in Minutes'].hist(bins=150, grid=False, color='red')
plt.title('Arrival Delay in Minutes Per Frequency')
plt.xlabel('Arrival Delay in Minutes', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()
#%%-------------------------------------------------------------------------------------
# selected categorical variables visualization
sns.catplot(x="Inflight wifi service", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Departure/Arrival time convenient", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Ease of Online booking", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Online boarding", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Food and drink", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Seat comfort", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Inflight entertainment", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Inflight service", kind="count", palette="Set1",data=df)
plt.show()
sns.catplot(x="Cleanliness", kind="count", palette="Set1",data=df)
plt.show()

#%%-------------------------------------------------------------------------------------
# selected features vs satisfaction comparison visualization
def boxplot(variable):
    sns.boxplot(x="satisfaction", y=df[variable], data=df)
    plt.show()
selected_var = ["Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking",
                "Food and drink","Online boarding","Seat comfort","Inflight entertainment","Inflight service","Cleanliness"]
for n in selected_var:
    boxplot(n)

#%%-------------------------------------------------------------------------------------
# Outlier detection #https://www.geeksforgeeks.org/interquartile-range-to-detect-outliers-in-data/

# Boxplot before removing outlier
sns.boxplot(df['Flight Distance'])
plt.title("Box Plot before outlier removing")
plt.show()
# Outlier detection using interquartile rule
def drop_outliers(df_1,var_1):
    iqr = 1.5 * (np.percentile(df_1[var_1],75) - np.percentile(df_1[var_1],25))
    df_1.drop(df_1[df_1[var_1] > (iqr + np.percentile(df_1[var_1],75))].index, inplace=True)
    df_1.drop(df_1[df_1[var_1] < (np.percentile(df_1[var_1],25) - iqr)].index, inplace=True)
# Remove outliers
drop_outliers(df,'Flight Distance')
# Boxplot after removing outlier
sns.boxplot(df['Flight Distance'])
plt.title("Box Plot after outlier removing")
plt.show()

sns.boxplot(df['Departure Delay in Minutes'])
plt.title("Box Plot before outlier removing")
plt.show()
drop_outliers(df,'Departure Delay in Minutes')
sns.boxplot(df['Departure Delay in Minutes'])
plt.title("Box Plot after outlier removing")
plt.show()

sns.boxplot(df['Arrival Delay in Minutes'])
plt.title("Box Plot before outlier removing")
plt.show()
drop_outliers(df,'Arrival Delay in Minutes')
sns.boxplot(df['Arrival Delay in Minutes'])
plt.title("Box Plot after outlier removing")
plt.show()

#%%-------------------------------------------------------------------------------------
# drop unnamed:0 and ID since the two features does not affect the target class
del df['Unnamed: 0']
del df['id']

#%%-------------------------------------------------------------------------------------
# Checking if the dataset is balanced or imbalanced
# Replace 0 and 1 in the target variable 'satisfaction'
df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)

plt.figure(figsize=(10,7))
df.satisfaction.value_counts(normalize=True).plot(kind='bar', color=['orange','blue'], alpha=.9, rot=0)
plt.title('neutral or dissatisfied(0) and satisfied(1)')
plt.show()

#%%-------------------------------------------------------------------------------------
# show correlation map
print(df.corr())
plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(),annot=True, cmap='Oranges_r')
plt.show()

#%%-------------------------------------------------------------------------------------
# split the dataset and encode the variables

# encoding the categorical features
categ_cols = ['Gender', 'Customer Type', 'Type of Travel',
       'Class', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness']
df[categ_cols]= df[categ_cols].apply(LabelEncoder().fit_transform)
X = df.values[:,:-1]

# encoding the class with sklearn's LabelEncoder
Y_data = df.values[:, -1]
class_le = LabelEncoder()
# fit and transform the class
y = class_le.fit_transform(Y_data)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#%%-------------------------------------------------------------------------------------
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
# Create the classifier object and performing training
clf_1 = LogisticRegression()
clf_1.fit(X_train, y_train)

# Make predicitions
y_pred_1 = clf_1.predict(X_test)
y_pred_score_1 = clf_1.predict_proba(X_test)

# calculate metrics
print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred_1))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred_1) * 100)
print("\n")

print("F1-score : ", f1_score(y_test, y_pred_1) * 100)
print("\n")


print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score_1[:,1]) * 100)
print("\n")

# plot AUC Curve
y_pred_proba_1 = clf_1.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba_1)
auc = roc_auc_score(y_test, y_pred_proba_1)

plt.plot(fpr,tpr,label="Logistic Reg AUC="+str(auc))
plt.legend(loc=4)
plt.show()

# confusion matrix for LR model
conf_matrix = confusion_matrix(y_test, y_pred_1)
class_names = df['satisfaction'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
# Show heat map
plt.tight_layout()
plt.show()

#%%-------------------------------------------------------------------------------------
# Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
# Create the classifier object and performing training
clf_2 = GaussianNB()
clf_2.fit(X_train, y_train)

# Make predicitions
y_pred_2 = clf_2.predict(X_test)
y_pred_score_2 = clf_2.predict_proba(X_test)

# calculate metrics
print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred_2))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred_2) * 100)
print("\n")

print("F1-score : ", f1_score(y_test, y_pred_2) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score_2[:,1]) * 100)
print("\n")

# plot AUC Curve
y_pred_proba_2 = clf_2.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba_2)
auc = roc_auc_score(y_test, y_pred_proba_2)

plt.plot(fpr,tpr,label="Naive Bayes AUC="+str(auc))
plt.legend(loc=4)
plt.show()

# confusion matrix for NB model
conf_matrix = confusion_matrix(y_test, y_pred_2)
class_names = df['satisfaction'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
# Show heat map
plt.tight_layout()
plt.show()

#%%-------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
clf_3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf_3.fit(X_train, y_train)

# Make predicitions
y_pred_3 = clf_3.predict(X_test)
y_pred_score_3 = clf_3.predict_proba(X_test)

# calculate metrics
print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred_3))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred_3) * 100)
print("\n")

print("F1-score : ", f1_score(y_test, y_pred_3) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score_3[:,1]) * 100)
print("\n")

# plot AUC Curve
y_pred_proba_3 = clf_3.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba_3)
auc = roc_auc_score(y_test, y_pred_proba_3)

plt.plot(fpr,tpr,label="Decision Tree AUC="+str(auc))
plt.legend(loc=4)
plt.show()

# confusion matrix for DT model
conf_matrix = confusion_matrix(y_test, y_pred_3)
class_names = df['satisfaction'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
# Show heat map
plt.tight_layout()
plt.show()
#%%-------------------------------------------------------------------------------------
# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Create the classifier object and performing training
clf_4 = RandomForestClassifier(n_estimators=100)
clf_4.fit(X_train, y_train)

#plot feature importances
# get feature importance
importances = clf_4.feature_importances_

# convert the importances into one-dimensional 1d array with corresponding df column names as axis labels
f_importances = pd.Series(importances, df.iloc[:,1:].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(20, 15), rot=90, fontsize=15)
# show the plot
plt.tight_layout()
plt.show()

#select features to perform training with random forest with k columns
# select the training dataset on k-features
newX_train = X_train[:, clf_4.feature_importances_.argsort()[::-1][:16]]


# select the testing dataset on k-features
newX_test = X_test[:, clf_4.feature_importances_.argsort()[::-1][:16]]

#perform training with random forest with k columns
# specify random forest classifier and train the model
clf_4_k_features = RandomForestClassifier(n_estimators=100)
clf_4_k_features.fit(newX_train, y_train)

# Make predicitions
y_pred_4 = clf_4.predict(X_test)
y_pred_score_4 = clf_4.predict_proba(X_test)

# prediction on test using k features
y_pred_4_k_features = clf_4_k_features.predict(newX_test)
y_pred_4_k_features_score = clf_4_k_features.predict_proba(newX_test)

# calculate metrics gini model
print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred_4))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_4) * 100)
print("\n")
print("F1-score : ", f1_score(y_test, y_pred_4) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score_4[:,1]) * 100)

# calculate metrics entropy model
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_4_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_4_k_features) * 100)
print("\n")
print("F1-score : ", f1_score(y_test, y_pred_4_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_4_k_features_score[:,1]) * 100)

# plot AUC Curve
y_pred_proba_4 = clf_4.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba_4)
auc = roc_auc_score(y_test, y_pred_proba_4)

plt.plot(fpr,tpr,label="Random Forest AUC="+str(auc))
plt.legend(loc=4)
plt.show()

# confusion matrix for RF model
conf_matrix = confusion_matrix(y_test, y_pred_4_k_features)
class_names = df['satisfaction'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
# Show heat map
plt.tight_layout()
plt.show()
#%%-------------------------------------------------------------------------------------
# KNN Model
# standardize the data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
stdsc = StandardScaler()
stdsc.fit(X_train)

X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

# perform training
# creating the classifier object and perform the train
clf_5 = KNeighborsClassifier(n_neighbors=5)
clf_5.fit(X_train_std, y_train)

# make predictions
# prediction on test
y_pred_5 = clf_5.predict(X_test_std)

# prediction probabilities on test
y_pred_score_5 = clf_5.predict_proba(X_test_std)

# calculate metrics
print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_5))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred_5) * 100)
print("\n")

print("F1-score : ", f1_score(y_test, y_pred_5) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score_5[:,1]) * 100)
print("\n")

# plot AUC Curve
y_pred_proba_5 = clf_5.predict_proba(X_test_std)[::,-1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba_5)
auc = roc_auc_score(y_test, y_pred_proba_5)

plt.plot(fpr,tpr,label="KNN AUC="+str(auc))
plt.legend(loc=4)
plt.show()

# confusion matrix for KNN model
conf_matrix = confusion_matrix(y_test, y_pred_5)
class_names = df['satisfaction'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
# Show heat map
plt.tight_layout()
plt.show()
#%%-------------------------------------------------------------------------------------
# Random forest model has the highest accuracy score 96%, F1-score 95%, and ROC_AUC score 99% when compared with other models.
from tabulate import tabulate
table = [['Logistic Regression', 82.84, 81.17, 88.89],
         ['Naive Bayes', 85.93, 83.93, 91.88],
         ['Decision Tree with Entropy', 94.27, 93.65, 94.23],
         ['Random Forest', 95.78, 95.21, 99.34],
         ['KNN', 92.22, 91.04, 96.58]]
print(tabulate(table, headers=['Model','Accuracy score','F1-score','ROC_AUC score']))