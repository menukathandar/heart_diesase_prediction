# Checking if there is any missing values
from load import load_dataset
dataset = load_dataset()
print(dataset.isna().sum())

# Dropping the columns which has missing values
dataset = dataset.dropna()
print(f'The length of the dataset after dropping missing values is {len(dataset)}')
