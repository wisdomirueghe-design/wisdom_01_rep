#Reading "iris" dataset and making sense of it
import pandas as pd

df = pd.read_csv('iris.csv', index_col= 0)

print(df.head(10))  #first 10 rows
print(df.tail())
print(df.shape)   #(150,6) 
print(df.info())
print(df.columns)   
print(df.describe())

print(df.loc[11:20])  #rows from index 11 to 20