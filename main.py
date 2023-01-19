from data_visualization import DataPlot
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


weight_tpr = 1
sourcedf = pd.read_csv('creditcard.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
sourcedf['Time'] = sourcedf['Time'].diff()
sourcedf['Time'][0] = 0
visualization = DataPlot()
visualization.boxplot(dataset=sourcedf, plot_name='Original boxplot', max_features_row=11)
visualization.binary_class_histogram(dataset=sourcedf, class_column_name='Class', plot_name='Original histogram',
                                     ncolumns=8)
X_train, X_test, y_train, y_test = train_test_split(sourcedf.iloc[:, :-1], sourcedf['Class'], test_size=0.2,
                                                    shuffle=True, stratify=sourcedf['Class'], random_state=0)
scaler = StandardScaler()
X_train_scaled = X_train
X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test_scaled = X_test
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
visualization.plot_output_class_distribution(y_train, y_test)

model = DummyClassifier(strategy='most_frequent')
model.fit(X_train_scaled, y_train)
print("Test score: {:.2f}".format(model.score(X_test_scaled, y_test)))
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_test, model.predict(X_test_scaled))))

algorithm = ['Logistic Regression', 'Naive Bayes', 'Decision Tree']
model = [LogisticRegression(random_state=0, C=0.1),
         GaussianNB(),
         DecisionTreeClassifier(random_state=0, max_depth=10)]
visualization.roc_curve_plot(algorithm, model, X_train_scaled, y_train, X_test_scaled, y_test, weight_tpr)
