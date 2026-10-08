from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

iris = load_iris()
x = iris.data  #features: sepal length, sepal width, petal length, petal width
y = iris.target  #target variables: 0(setosa), 1(versicolor), 2(virginica)

#splitting datasets into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42, stratify= y)
#decision classifier
d_class = DecisionTreeClassifier(random_state= 42)
d_class.fit(x_train, y_train)
#predictions
y_pred = d_class.predict(x_test)
#detailed metrics
print('classification report: ')
print(classification_report(y_test, y_pred, target_names = iris.target_names))

#Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average= 'weighted')
recall = recall_score(y_test, y_pred, average= 'weighted')

print(f'Accuracy: ', accuracy)  #0.933
print(f'Precision: ', precision)  #0.933
print(f'Recall: ', recall)  #0.933



