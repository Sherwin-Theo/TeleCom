import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

data1 = pd.read_csv('Telecom Customers Churn.csv')
data = data1.copy()

y = data['Churn']
X = data.drop(['customerID', 'Churn', 'TotalCharges'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

cat_cols_train = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
label_y_train = y_train.copy()
label_y_valid = y_valid.copy()

label_X_train = pd.get_dummies(label_X_train, dtype=int, sparse=False)
label_X_valid = pd.get_dummies(label_X_valid, dtype=int, sparse=False)
label_y_train = pd.get_dummies(label_y_train, dtype=int, sparse=False)
label_y_valid = pd.get_dummies(label_y_valid, dtype=int, sparse=False)

label_X_train, label_X_valid = label_X_train.align(X_valid, join='left', axis=1)

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=500,
                      learning_rate=0.1,
                      eval_metric="mae",
                      early_stopping_rounds=5)
model.fit(label_X_train, label_y_train,
          eval_set=[(label_X_valid, label_y_valid)],
          verbose=False)
pred = model.predict(label_X_valid)


test_score = accuracy_score(pred, label_y_valid)
print("Test score:", np.round(test_score,2))

confusion_matrix = metrics.confusion_matrix(label_y_valid.values.argmax(axis=1), pred.argmax(axis=1))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()
