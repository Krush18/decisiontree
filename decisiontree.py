# -*- coding: utf-8 -*-

"Program to run logistic regression to calculate customer churn using Watson telecom dataset"
##Importing require libraries for data preprocessing and logistic regression
import pandas as pd #For Linear Algebra
import numpy as np #For Visualizations
import matplotlib.pyplot as plt #For Visualizations
import seaborn as sns # For data exploration
#For Regression
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,\
mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
from math import sqrt

#Reading data from local directory
telecom_df = pd.read_csv("Telco-Customer-Churn.csv")
telecom_df.head()
#Next step is Data Exploration
telecom_df.info() # Dataset has no missing values
#checking count of unique values for all columns
telecom_df.nunique()
#Checking data distribution of all the columns (X variables and Y variable)telecom_df
#Data manipulations to clean the dataset. Let's start with the column TotalCharges
telecom_df["TotalCharges"] = telecom_df["TotalCharges"].replace(" ",np.nan) 
telecom_df["TotalCharges"] = telecom_df["TotalCharges"].astype(float)
telecom_df = telecom_df[telecom_df["TotalCharges"].notnull()]
telecom_df = telecom_df.reset_index()[telecom_df.columns]
# Next we will clean the data for column 'MultipleLines'
telecom_df["MultipleLines"] = telecom_df["MultipleLines"].replace({"No phone service":"No"}) 
telecom_df["MultipleLines"].unique()
cols_list = ["OnlineSecurity", "OnlineBackup", "DeviceProtection","TechSupport","StreamingTV", "StreamingMovies"]

for i in cols_list:
    sns.countplot(data = telecom_df, x = i)
    plt.show()
#As we see that, the columns from cols_list could be cleaned to have only 2 values each which are 'Yes' & 'No'
for i in cols_list:
    telecom_df[i] = telecom_df[i].replace({"No internet service":"No"})
#next we will clean tenure column - Currently tenure is integer column we can create seperate bins for the tenure and make a categorical column 
print(telecom_df["tenure"].unique())
plt.hist(telecom_df['tenure'])
plt.show()
# Function to create categorical column for tenure
def tenure_cat(telecom_df):
    if telecom_df["tenure"] <= 12:
        return "tenure-0-12"
    elif (telecom_df["tenure"] > 12) & (telecom_df["tenure"] <= 24):
        return "tenure-12-24"
    elif (telecom_df["tenure"] > 24) & (telecom_df["tenure"] <= 48):
        return "tenure-24-48"
    elif (telecom_df["tenure"] > 48) & (telecom_df["tenure"] <= 60):
        return "tenure-48-60"
    elif (telecom_df["tenure"] > 60):
        return "tenure-morethan-60"

telecom_df["tenure"] = telecom_df.apply(lambda telecom_df:tenure_cat(telecom_df),axis = 1)

#We can seperate the categorical columns, numerical columns and target column for data preprocessing
cust_Id = ['customerID']
target = ["Churn"]
colums_list   = telecom_df.nunique()[telecom_df.nunique() < 6].keys().tolist() #As there are 5 unique values in tenure column
cat_col_list   = [x for x in colums_list if x not in target + ["SeniorCitizen"]]
num_col_list   = [x for x in telecom_df.columns if x not in cat_col_list + target + cust_Id + ["SeniorCitizen"]]
print(cat_col_list,num_col_list,sep="\n")#Verify the categorical columns list and numerical columns list
#We will create encoding lables for the column Churn which is also our target column
labEncode = LabelEncoder()
telecom_df[target] = labEncode.fit_transform(telecom_df[target])
print(telecom_df['Churn'].unique())
#As we have created a list for all categorical columns, we can use it to create dummy variables for these relevant columns

telecom_df = pd.get_dummies(data = telecom_df,columns = cat_col_list)

print(telecom_df.info())
#Let's check the distribution of two numerical columns MonthlyCharges & TotalCharges against churn
churn_yes = telecom_df[telecom_df["Churn"]==1]
churn_no = telecom_df[telecom_df["Churn"] == 0]
sns.distplot(churn_yes["MonthlyCharges"])
sns.distplot(churn_yes["TotalCharges"])
#As the distribution of 'MonthlyCharges' and 'TotalCharges' looks skewed for both Churned and not churned customers
#Let's try to transform these variables.
churn_yes['Log_MonthlyCharges'] = churn_yes['MonthlyCharges'].apply(np.log)
sns.distplot(churn_yes["Log_MonthlyCharges"])
plt.show()
churn_yes['Log_TotalCharges'] = churn_yes['TotalCharges'].apply(np.log)
sns.distplot(churn_yes["Log_TotalCharges"])
plt.show()

telecom_train, telecom_test = train_test_split(telecom_df,test_size = 0.30,random_state = 100)
#print(telecom_df.info())

train_X = telecom_train[[i for i in telecom_df.columns if i not in cust_Id + target ]]
print(train_X.info())
train_Y = telecom_train[target]
test_X = telecom_test[[i for i in telecom_df.columns if i not in cust_Id + target ]]
test_Y = telecom_test[target]
print(test_Y['Churn'].unique())
#Next step is to fit the decision tree
dtc = DecisionTreeClassifier(criterion='gini',max_depth= 8,min_samples_split=10,min_samples_leaf=10)
dtc.fit(train_X,train_Y)
features = train_X.columns.values
predict_train = dtc.predict(train_X)
predict_test = dtc.predict(test_X)
max_depth = dtc.max_depth
print("{:.<23s}{:>10s}{:>15s}".format('\nDecisionTree Metrics','Training', 'Test'))
print("{:.<23s}{:9d}{:15d}".format('Observations',train_X.shape[0], test_X.shape[0]))
print("{:.<23s}{:>9s}{:>15s}".format('Split Criterion',dtc.criterion, dtc.criterion))
print("{:.<23s}{:9d}{:15d}".format('Max Depth',max_depth, max_depth))
print("{:.<23s}{:9d}{:15d}".format('Minimum Split Size',dtc.min_samples_split, dtc.min_samples_split))
print("{:.<23s}{:9d}{:15d}".format('Minimum Leaf  Size',dtc.min_samples_leaf, dtc.min_samples_leaf))

R2_train = r2_score(train_Y, predict_train)
R2_test = r2_score(test_Y, predict_test)
mse_train = mean_absolute_error(train_Y,predict_train) 
mse_test = mean_absolute_error(test_Y,predict_test) 
ase_train = mean_squared_error(train_Y,predict_train) 
ase_test = mean_squared_error(test_Y,predict_test) 

print("{:.<23s}{:9.4f}{:15.4f}".format('R-Squared', R2_train, R2_test))
print("{:.<23s}{:9.4f}{:15.4f}".format('Mean Absolute Error',mse_train, mse_test))
print("{:.<23s}{:9.4f}{:15.4f}".format('Average Sqaured Error',ase_test, ase_test))
print("{:.<23s}{:9.4f}{:15.4f}".format('Square Root ASE', sqrt(mse_train),sqrt(mse_test)))

