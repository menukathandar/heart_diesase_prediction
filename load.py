#Loading the dataset
import pandas as pd
dataset = pd.read_csv('heart_disease.csv')
print(f'The length of the dataset is {len(dataset)}')
print(dataset.head())