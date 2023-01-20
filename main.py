import pandas as pd
from utils import DataTools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix


weight_tpr = 1
pd.set_option('display.max_columns', None)
sourcedf = pd.read_csv('creditcard.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
sourcedf['Time'] = sourcedf['Time'].diff()
sourcedf['Time'][0] = 0
utils = DataTools()
utils.boxplot(dataset=sourcedf, plot_name='Original boxplot', max_features_row=11)
utils.binary_class_histogram(dataset=sourcedf, class_column_name='Class', plot_name='Original histogram', ncolumns=8)

X_train, X_test, y_train, y_test = train_test_split(sourcedf.iloc[:, :-1], sourcedf['Class'], test_size=0.2,
                                                    shuffle=True, stratify=sourcedf['Class'], random_state=0)
scaler = StandardScaler()
X_train_scaled = X_train
X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test_scaled = X_test
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
utils.plot_output_class_distribution(y_train, y_test)

model = DummyClassifier(strategy='most_frequent')
model.fit(X_train_scaled, y_train)
print("Test score: {:.2f}".format(model.score(X_test_scaled, y_test)))
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_test, model.predict(X_test_scaled))))

# Examples
algorithm = ['logistic regression', 'naive bayes', 'tree']
params = [{'C': 0.1}, {}, {'max_depth': 10}]
utils.roc_curve_plot(algorithm, params, X_train_scaled, y_train, X_test_scaled, y_test, weight_tpr, 'Example')

# Grid search and model optimization
algorithm = ['logistic regression', 'linear svc', 'naive bayes', 'tree']
scale = [''] * len(algorithm)
scoring = 'roc_auc'
params = [
    {'classifier': [], 'preprocessing': [], 'classifier__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]},
    {'classifier': [], 'preprocessing': [], 'classifier__C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]},
    {'classifier': [], 'preprocessing': []},
    {'classifier': [], 'preprocessing': [], 'classifier__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
grid = utils.cross_grid_validation(algorithm, scale, params, X_train_scaled, y_train, X_test_scaled, y_test, scoring, 5)
pd_grid = pd.DataFrame(grid.cv_results_)
utils.param_sweep_plot(algorithm, params=pd_grid['params'], test_score=pd_grid['mean_test_score'])
params = [{'C': 0.01}, {'C': 0.0001}, {}, {'max_depth': 3}]
utils.roc_curve_plot(algorithm, params, X_train_scaled, y_train, X_test_scaled, y_test, weight_tpr, 'Optimal')

algorithm = ['tree', 'tree', 'tree']
params = [{'max_depth': 20}, {'max_depth': 10}, {'max_depth': 3}]
utils.roc_curve_plot(algorithm, params, X_train_scaled, y_train, X_test_scaled, y_test, weight_tpr, 'Tree')

algorithm = ['logistic regression', 'logistic regression', 'logistic regression']
params = [{'C': 1}, {'C': 0.1}, {'C': 0.01}]
utils.roc_curve_plot(algorithm, params, X_train_scaled, y_train, X_test_scaled, y_test, weight_tpr, 'Logreg')
