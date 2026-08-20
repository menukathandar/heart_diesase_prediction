#Loading the dataset
import pandas as pd
def load_dataset():
    return pd.read_csv('heart_disease.csv')
dataset = load_dataset()
print(f'The length of the dataset is {len(dataset)}')
print(dataset.head())