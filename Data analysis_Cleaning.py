import pandas as pd

df = pd.read_csv('iris.csv', index_col= 0)

#detecting missing value in the file
print(df.isnull())   #all false
'''''
there are no missing values in iris.csv
dropping or filling of any data column would not be needed
'''''
#detecting duplicate values
print(df.duplicated())    #All false, no duplicates

#filtering dataset to rows of (iris-setosa and Sepallength <= 5.0)
filt_df = df[(df['SepalLengthCm'] <= 5.0) & (df['Species'] == 'Iris-setosa')]

#Adding a new column to state size of sepal of filtered dataset
filt_df['Size'] = filt_df['SepalLengthCm'].apply(
    lambda s: 'small sepal' if s < 4.5 else 'Big sepal'
)
print(filt_df)

#Creating a new column in the dataset(App. sepal Area)
df['Sepal Area'] = df['SepalLengthCm'] * df['SepalWidthCm']
print(df)

#Calculating mean of petal width categorized by the value of the petal length
group_df = df.groupby('PetalLengthCm')['PetalWidthCm'].mean()
print(group_df)  

