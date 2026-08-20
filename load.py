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