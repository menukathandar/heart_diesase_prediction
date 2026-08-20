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
