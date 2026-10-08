import seaborn as sns
import matplotlib.pyplot as plt

#Exploratory Data analysis
iris = sns.load_dataset('iris')
print(iris.head())
print(iris.tail())
print(iris.shape)
print(iris.corr())
print(iris.isnull().sum())  #number of missing values
print(iris.info())  #Dataset info
print(iris['species'].value_counts())  #number of valid objects in each column
print(iris.duplicated().sum())  #number of duplicate values

#DATA VISUALIZARTION PLOTS
print(sns.histplot(iris, x = 'sepal_length'))  #histogram
plt.show()
#variation of 'sepal_width' across species using boxplot
print(sns.boxplot(iris, x = 'sepal_width', color= 'red'))
plt.show()
#variation of 'sepal_width across species using kde plot
print(sns.displot(iris, x = 'sepal_width', kind='kde' , col= 'species'))
plt.show()
#pairplot for all numerical columns coloured by species
print(sns.pairplot(iris, hue= 'species'))
plt.show()
#correlation heatmap for numerical columns
correlate = iris.corr(numeric_only = True)
print(sns.heatmap(correlate, annot= True, fmt= '.2f', cmap = 'coolwarm'))
plt.show()
#bar plot showing mean petal length of each species
print(sns.barplot(iris, x= 'species', y= 'petal_length', ci= None))
plt.title('Average petal length by species')
plt.show()

#summary statistics
summary = iris.describe()
print(summary)  #overall summary(mean,count,standard dev,...)
#species mean
print(iris.groupby('species').mean())
#species individual median
print(iris.groupby('species').median())
#summary statistics for each individual species
print(iris.groupby('species').describe())