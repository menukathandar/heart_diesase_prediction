#Loading the dataset
import pandas as pd
dataset = pd.read_csv('heart_disease.csv')
print(f'The length of the dataset is {len(dataset)}')
print(dataset.head())

#Checking if there is any missing values
print(dataset.isna().sum())

#Dropping the columns with missing values
dataset = dataset.dropna()
print(f'The length of the dataset after dropping the missing values is {len(dataset)}')

#Checking the variable data types
print(dataset.dtypes)

# Number of vessels and Thal is represented as float but it should be int
cols = ['Number of major vessels', 'thal']
dataset[cols] = dataset[cols].astype(int)

#checking the datatype after changing from float to int
print(dataset.dtypes)

#Checking for any duplicate rows
print(dataset.duplicated().any())
# Since the output is false, so there are no any duplicate rows

#Checking value count for categorical variables
print(dataset['Sex'].value_counts(),'\n')
print(dataset['Chest Pain Type'].value_counts(),'\n')
print(dataset['Fasting Blood Sugar'].value_counts(),'\n')

# SInce Sex and Fasting Blood SUgar are binary variables, so we can use 0 and 1 to replace them
#For variable sex: 1 = female,  0 = male
#For Fasting Blood SUgar: 1 = True, 0 = False
#Based on domain expert advice, we can use following rule to transform the chest pain type
# Value 1 = typical angina
# Value 2 = atypical angina
# VAlue 3 = non-anginal pain
# Value 4 = asymptomatic

dataset['Sex'] = dataset['Sex'].replace({'female': 1, 'male': 0})
dataset['Fasting Blood Sugar'] = dataset['Fasting Blood Sugar'].replace({True: 1, False: 0})
dataset['Chest Pain Type'] = dataset['Chest Pain Type'].replace({'typical angina': 1, 'atypical angina': 2, 'non-anginal pain': 3, 'asymptomatic': 4})
print(dataset.head())

#Checking dataset shape to make sure we have the same no of columns and rows as expected
print(dataset.shape)

#Defining input variables and the target variable
x = dataset.drop('Diagnosis', axis = 1) #axis = 1 means drop row not column
y = dataset['Diagnosis']

#Quick check
print(x.shape) # Output is (297, 13)
print(y.shape) # Output is (297, )

#Splitting the dataset and normalisaing the data
#(Using 10% of dataset for testing with a random state of 1 )
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=1)

#Applying normalisation on both training and testing dataset
from sklearn.preprocessing import MinMaxScaler

#fitting scalar on training data
norm = MinMaxScaler().fit(x_train)

#transforming training data
x_train_norm = norm.transform(x_train)

#transforming testing data
x_test_norm = norm.transform(x_test)

#Training a model with logistic regression and SVM for classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#logistic regression model
model = LogisticRegression(solver="liblinear")
model.fit(x_train_norm, y_train)
test_score = model.score(x_test_norm, y_test)
print(f'Testing accuracy of LR: {test_score}')

#Support Vector Classifier
model = SVC()
model.fit(x_train_norm, y_train)
test_score = model.score(x_test_norm, y_test)
print(f'Testing accuracy of SVC: {test_score}')

#The outputs are : Testing accuracy of LR: 0.9666666666666667
#Testing accuracy of SVC: 0.8666666666666667

#COnclusion regarding this: LR got about 96.7% of predictions correct and SVC got about 86.7% correct. So, in this scenario, LR performed better.

#training a model with 5-fold cross validation
#setting random state as 2
from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5, shuffle = True, random_state = 2)

#Getting the average accuracy scores based on the cross validation results and evaluate both models on the testing dataset
from sklearn.model_selection import cross_val_score
model = LogisticRegression(solver = 'liblinear')
results = cross_val_score(model, x_train_norm, y_train, cv = kfold)
print(f'Average accuracy of LR is {results.mean()}')

model = SVC()
results = cross_val_score(model, x_train_norm, y_train, cv = kfold)
print(f'Average accuracy of SVM is {results.mean()}')

#The output is Average accuracy of LR is 0.8389937106918239
#Average accuracy of SVM is 0.8089447938504544
print(results)