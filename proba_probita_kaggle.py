import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import warnings
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Ignore warnings
warnings.filterwarnings("ignore")
#Data upload
df = pd.read_csv("D:\python\WELFARE.csv")
X = df[['gender', 'age', 'age2', 'married', 'schooling',  'entrepreneur', 'employee',  'city', 'health']]
y = df[['much_worse']]
# Splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Running XGB model
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',
                              n_jobs=1,eval_metric='logloss').fit(X_train, y_train.values.ravel())
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, xgb_model.predict(X_test))))
#Plotting predictor importnace
xgb.plot_importance(xgb_model, height=0.5)
plt.yticks(fontsize = 10)
plt.ylabel('Predictors', fontsize = 10)
plt.title('Predictor importance', fontsize = 12)
plt.show()
#Decision trees
dectree_basic = DecisionTreeClassifier()
dectree_basic.max_depth = 100
dectree_basic.fit(X_train,y_train.values.ravel())
y_pred = dectree_basic.predict(X_test)
print('Decision tree accuracy score: {0:0.4f}'. format(accuracy_score(y_test,y_pred)))

# Further steps may be done later

   
