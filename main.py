import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score


#### read file ###
input_file_missing = "allallTrain.txt"
df_missing = pd.read_csv(input_file_missing, header = 0, sep='\t')
#### separate test and train datai, and fill in the blanks values ###
data_x_miss = df_missing.ix[:,:-1]
data_y_miss = df_missing.T.iloc[-1]
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data_x_miss)
data_X_miss = imp.transform(data_x_miss)

X_train , X_test, y_train, y_test = train_test_split (data_X_miss, data_y_miss, test_size=0.3, random_state = 0)

#### Preprocessing ###
scale_all = scale(data_X_miss)
scale_train = scale(X_train)
scale_test = scale(X_test)

### LinearRegression ###
regr = linear_model.LinearRegression(normalize = True)
regr.fit(scale_train, y_train)
accura = regr.score(scale_test, y_test)
print ("SCORE LINEAR REGRESSION: ", accura)
regr_cross = linear_model.LinearRegression(normalize = True)
regr_cross.fit(scale_all, data_y_miss)
predi = cross_val_predict(regr_cross, scale_all, data_y_miss, cv=10)
scorePre = regr_cross.score(scale_all, data_y_miss)
print ("SCORE LINEAR REGRESSION CrossValidation: ", scorePre)

#### SVR kernel = rbf #######

rbf = SVR(kernel = 'rbf', C =1e3, gamma=0.1)
rbf.fit(scale_train, y_train)
accura = rbf.score(scale_test, y_test)
print ("SCORE SVR with rbf: ", accura)
rbf_cross = SVR(kernel = 'rbf', C =1e3, gamma=0.1)
predi = cross_val_predict(rbf, scale_all, data_y_miss, cv=10)
scorePre = rbf.score(scale_all, data_y_miss)
print ("SCORE SVR with rbf CrossValidation: ", scorePre)
rbf_cross.fit(scale_train, y_train)

#### SVR kernel = linear #######

model_linear = SVR(kernel = 'linear', C =0.02)
model_linear.fit(scale_train, y_train) 
accura = model_linear.score(scale_test, y_test)
print ("SCORE SVR with linear: ", accura)
model_linear_cross = SVR(kernel = 'linear', C =0.02)
predi = cross_val_predict(model_linear, scale_all, data_y_miss, cv=10)
scorePre = model_linear.score(scale_all, data_y_miss)
print ("SCORE SVR with Linear CrossValidation: ", scorePre)


#### Predict results ####
result = "output.txt"
data_x_miss = pd.read_csv(result, header = 0, sep='\t')
#### separate test and train datai, and fill in the blanks values ###
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data_x_miss)
data_X_miss = imp.transform(data_x_miss)
result_vector = rbf_cross.predict(data_X_miss)

f = open('predict_file.txt', 'w')

for el in result_vector:
  f.write("%s\n	" % el)
