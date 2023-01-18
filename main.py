from data_visualization import DataPlot
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


sourcedf = pd.read_csv('creditcard.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
scaler = StandardScaler()
sourcedf['Time'][1:] = [sourcedf['Time'][x] - sourcedf['Time'][x - 1] for x in range(1, sourcedf['Time'].shape[0])]
df = scaler.fit_transform(sourcedf[['Time']])
sourcedf['Time'] = df
df = scaler.fit_transform(sourcedf[['Amount']])
sourcedf['Amount'] = df
visualization = DataPlot()
visualization.boxplot(dataset=sourcedf, plot_name='Original boxplot', max_features_row=11)
visualization.binary_class_histogram(dataset=sourcedf, class_column_name='Class', plot_name='Original histogram',
                                     ncolumns=8)

X_train, X_test, y_train, y_test = train_test_split(sourcedf.iloc[:, :-1], sourcedf['Class'], test_size=0.2,
                                                    shuffle=True, stratify=sourcedf['Class'], random_state=0)
visualization.plot_output_class_distribution(y_train, y_test)

model_dummy = DummyClassifier(strategy='most_frequent')
model_dummy.fit(X_train, y_train)
pred_most_frequent = model_dummy.predict(X_test)
print("Unique predicted labels: {}".format(np.unique(pred_most_frequent)))
print("Test score: {:.2f}".format(model_dummy.score(X_test, y_test)))
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_test, pred_most_frequent)))

model_logreg = LogisticRegression(C=0.1)
model_logreg.fit(X_train, y_train)
pred_logreg = model_logreg.predict(X_test)
print("logreg score: {:.2f}".format(model_logreg.score(X_test, y_test)))
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_test, pred_logreg)))
