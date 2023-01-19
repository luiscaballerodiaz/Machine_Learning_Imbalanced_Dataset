from data_visualization import DataPlot
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.cm as cm


weight_tpr = 0.5
weight_fpr = 1 - weight_tpr

sourcedf = pd.read_csv('creditcard.csv')
print("Full source data from CSV type: {} and shape: {}".format(type(sourcedf), sourcedf.shape))
sourcedf['Time'][1:] = [sourcedf['Time'][x] - sourcedf['Time'][x - 1] for x in range(1, sourcedf['Time'].shape[0])]

visualization = DataPlot()
visualization.boxplot(dataset=sourcedf, plot_name='Original boxplot', max_features_row=11)
visualization.binary_class_histogram(dataset=sourcedf, class_column_name='Class', plot_name='Original histogram',
                                     ncolumns=8)
X_train, X_test, y_train, y_test = train_test_split(sourcedf.iloc[:, :-1], sourcedf['Class'], test_size=0.2,
                                                    shuffle=True, stratify=sourcedf['Class'], random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
visualization.plot_output_class_distribution(y_train, y_test)

model = DummyClassifier(strategy='most_frequent')
model.fit(X_train_scaled, y_train)
print("Test score: {:.2f}".format(model.score(X_test_scaled, y_test)))
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_test, model.predict(X_test_scaled))))

algorithm = ['Logistic Regression', 'Naive Bayes', 'Decision Tree']
model = [LogisticRegression(random_state=0, C=0.1),
         GaussianNB(),
         DecisionTreeClassifier(random_state=0, max_depth=10)]
cmap = cm.get_cmap('Set1')
colors = cmap.colors
for i in range(len(model)):
    model[i].fit(X_train_scaled, y_train)
    print("Test score {}: {:.2f}".format(algorithm[i], model[i].score(X_test_scaled, y_test)))
    y_pred_prob = model[i].predict_proba(X_test_scaled)[:, 1]
    print('AUC {}: {}'.format(algorithm[i], roc_auc_score(y_test, y_pred_prob)))
    fpr, tpr, th = roc_curve(y_test, y_pred_prob)
    youden = weight_tpr * tpr - weight_fpr * fpr
    index_opt = np.argmax(youden)
    index05 = np.argmin(np.abs(th - 0.5))
    plt.plot(fpr, tpr, color=colors[i], linewidth=2, label='ROC curve ' + algorithm[i])
    plt.scatter(fpr[index05], tpr[index05], s=200, marker='o', label='Default threshold ' + algorithm[i],
                edgecolor='k', lw=2, color=colors[i])
    plt.scatter(fpr[index_opt], tpr[index_opt], s=200, marker='^', label='Optimal threshold ' + algorithm[i],
                edgecolor='k', lw=2, color=colors[i])
    print("Confusion matrix:\n{}\n".format(confusion_matrix(y_test, model[i].predict(X_test_scaled))))
plt.title('Receiver Operating Characteristics (ROC) curve with TPR weight ' + str(weight_tpr), fontsize=24)
plt.xlabel("FPR (false positives divided by negative samples)", fontsize=14)
plt.ylabel("TPR (true positives divided by positive samples)", fontsize=14)
plt.grid()
plt.legend()
plt.savefig('ROC curve TPR weight ' + str(weight_tpr) + '.png', bbox_inches='tight')
plt.clf()







