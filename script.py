#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:33:45 2020

@author: damienchambon
All rights reserved
"""

############################
## IMPORTER LES LIBRARIES ##
############################

import pandas as pd
import numpy as np
pd.options.display.max_columns = None # afficher toutes les colonnes du dataset



##########################
## IMPORTER LES DONNEES ##
##########################

df = pd.read_csv('telco.csv')
df

df.head()
df.info()

df['gender']
df['customerID']

df[df['MonthlyCharges']<30]['gender']



######################################
## TRANSFORMER LE TYPE DES COLONNES ##
######################################

df = df.astype({'gender':'category'})

list_columns = list(df)
list_columns

for col in list_columns:
    if col not in ('customerID','gender','tenure','MonthlyCharges','TotalCharges'):
        # pour chaque colonne qui n'est pas dans la liste, la transformer en type catégorie
        df = df.astype({col:'category'})
        
#df = df.astype({'TotalCharges':'float64'}) # pas possible car il y a une valeur ' '

df['TotalCharges'] = df['TotalCharges'].replace(' ',np.nan) # enlever cette valeur

df = df.astype({'TotalCharges':'float64'})

df.info()

df['Dependents']



##############################
## ISOLER RESPONSE VARIABLE ##
##############################

from sklearn.model_selection import train_test_split # pip install scikit-learn

column_to_predict = 'Dependents' # commenter cette ligne si clustering

df_cleaned = df[df[column_to_predict].notna()] # commenter cette ligne si clustering

train_set, test_set = train_test_split(df_cleaned, test_size=0.1, random_state=42) # commenter cette ligne si clustering
train_set, test_set = train_test_split(df, test_size=0.1, random_state=42) # commenter cette ligne si régression ou classification

df_cleaned.info() # commenter cette ligne si clustering
train_set.info()
test_set.info()


response_train = train_set[column_to_predict] # commenter cette ligne si clustering
train_set = train_set.drop([column_to_predict], axis=1) # commenter cette ligne si clustering

response_test = test_set[column_to_predict] # commenter cette ligne si clustering
test_set = test_set.drop([column_to_predict], axis=1) # commenter cette ligne si clustering



############################################
## PIPELINES POUR TRANSFORMER LES DONNEES ##
############################################

cat_attribs_df = train_set.select_dtypes(include=['category'])
cat_attribs_df
cat_attribs = list(cat_attribs_df)
cat_attribs

num_attribs_df = train_set.select_dtypes(include=['int64','float64'])
num_attribs_df
num_attribs = list(num_attribs_df)
num_attribs

len(num_attribs) + len(cat_attribs)
len(list(train_set))


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

full_pipeline.fit(train_set)
train_set_prepared = full_pipeline.transform(train_set)
train_set_prepared[0] # les valeurs des variables ne sont plus identifiables car transformées



#########################
## REGRESSION LINEAIRE ##
#########################

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(train_set_prepared,response_train)

response_test_part = response_test.iloc[:5]
test_set_part = test_set.iloc[:5]

test_set_part_prepared = full_pipeline.transform(test_set_part)

lin_reg.predict(test_set_part_prepared)
response_test_part


test_set_prepared = full_pipeline.transform(test_set)

pred = lin_reg.predict(test_set_prepared)

from sklearn.metrics import mean_squared_error
import numpy as np

mean_squared_error(response_test,pred)

np.sqrt(mean_squared_error(response_test,pred))

lin_reg.coef_



########################################
## REGRESSION LINEAIRE REGULARISATION ##
########################################

from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=1.0,l1_ratio=0.5)

elastic_net.fit(train_set_prepared,response_train)

elastic_net.predict(test_set_part_prepared)
response_test_part

pred = elastic_net.predict(test_set_prepared)

np.sqrt(mean_squared_error(response_test,pred))


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'l1_ratio': np.arange(0, 1, 0.05),'alpha': np.arange(0.001, 5, 0.5)}
  ]
ela_reg = ElasticNet()
grid_search = GridSearchCV(ela_reg, param_grid, cv=5,
                           scoring='neg_root_mean_squared_error',n_jobs=-1)
grid_search.fit(train_set_prepared, response_train)

grid_search.best_params_
final_model = grid_search.best_estimator_

final_predictions = final_model.predict(test_set_prepared)
final_mse = mean_squared_error(response_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse



###########################
## REGRESSION LOGISTIQUE ##
###########################

from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression()

logistic_reg.fit(train_set_prepared,response_train)

logistic_reg.classes_

response_test_part = response_test.iloc[:5]
test_set_part = test_set.iloc[:5]

test_set_part_prepared = full_pipeline.transform(test_set_part)

logistic_reg.predict(test_set_part_prepared)
response_test_part

logistic_reg.predict_proba(test_set_part_prepared[1].reshape(1,-1))


test_set_prepared = full_pipeline.transform(test_set)

pred = logistic_reg.predict(test_set_prepared)


from sklearn.metrics import confusion_matrix

confusion_matrix(response_test, pred,labels=["Yes", "No"])

tp, fn, fp, tn = confusion_matrix(response_test, pred,labels=["Yes", "No"]).ravel()

fn

precision = tp/(tp+fp)
precision

recall = tp/(tp+fn)
recall



############################
## SUPPORT VECTOR MACHINE ##
############################

from sklearn.svm import SVC

svm = SVC()

svm.fit(train_set_prepared,response_train)

predictions= svm.predict(test_set_prepared)

confusion_matrix(response_test, predictions, labels=["Yes", "No"])

tp, fn, fp, tn = confusion_matrix(response_test, predictions, labels=["Yes", "No"]).ravel()

precision = tp/(tp+fp)
precision
recall = tp/(tp+fn)
recall

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'C': np.arange(1, 10, 0.5)}
  ]
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5,scoring='recall',n_jobs=-1)
grid_search.fit(train_set_prepared, response_train)


from sklearn.metrics import recall_score, make_scorer
recall_scorer = make_scorer(score_func=recall_score,pos_label="Yes")

grid_search = GridSearchCV(svm, param_grid, cv=5,scoring=recall_scorer,n_jobs=-1)
grid_search.fit(train_set_prepared, response_train)

grid_search.best_params_
final_model = grid_search.best_estimator_

final_predictions = final_model.predict(test_set_prepared)

confusion_matrix(response_test, final_predictions, labels=["Yes", "No"])

tp, fn, fp, tn = confusion_matrix(response_test, final_predictions, labels=["Yes", "No"]).ravel()

precision = tp/(tp+fp)
precision
recall = tp/(tp+fn)
recall



########################
## ARBRES DE DECISION ##
########################

from sklearn.tree import DecisionTreeClassifier

test_set_prepared = full_pipeline.transform(test_set)

tree1 = DecisionTreeClassifier()
tree1.fit(train_set_prepared, response_train)
pred1 = tree1.predict(test_set_prepared)

tree2 = DecisionTreeClassifier(max_depth = 50)
tree2.fit(train_set_prepared, response_train)
pred2 = tree2.predict(test_set_prepared)

tree3 = DecisionTreeClassifier(min_samples_split=10)
tree3.fit(train_set_prepared, response_train)
pred3 = tree3.predict(test_set_prepared)

tree4 = DecisionTreeClassifier(max_depth = 50,min_samples_split=10)
tree4.fit(train_set_prepared, response_train)
pred4 = tree4.predict(test_set_prepared)

count = 0

for index in range(len(pred1)):
    if not ((pred1[index]==pred2[index]) and (pred2[index]==pred3[index]) and (pred3[index]==pred4[index])):
        count += 1

count
len(pred1)

from sklearn.metrics import confusion_matrix

tp, fn, fp, tn = confusion_matrix(response_test, pred4, labels=["Yes", "No"]).ravel()

precision = tp/(tp+fp)
precision
recall = tp/(tp+fn)
recall



###########################
## METHODES ENSEMBLISTES ##
###########################

from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

bagging = BaggingClassifier(LogisticRegression(), max_samples=0.8, max_features=0.8)

bagging.fit(train_set_prepared, response_train)
test_set_prepared = full_pipeline.transform(test_set)
predictions= bagging.predict(test_set_prepared)

from sklearn.metrics import confusion_matrix
tp, fn, fp, tn = confusion_matrix(response_test, predictions, labels=["Yes", "No"]).ravel()

precision = tp/(tp+fp)
precision
recall = tp/(tp+fn)
recall

#######

from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(n_estimators=500)
rand_forest.fit(train_set_prepared, response_train)
predictions= rand_forest.predict(test_set_prepared)

tp, fn, fp, tn = confusion_matrix(response_test, predictions, labels=["Yes", "No"]).ravel()

precision = tp/(tp+fp)
precision
recall = tp/(tp+fn)
recall

#######

from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression()
clf2 = SVC(C=2.5)
clf3 = SVC(C=1)
clf4 = DecisionTreeClassifier(max_depth=100)
clf5 = DecisionTreeClassifier()

voting_class = VotingClassifier(estimators=[('lr', clf1), ('svc1', clf2), ('svc2', clf3), ('dtc1', clf4), ('dtc2', clf5)],voting='hard')

voting_class.fit(train_set_prepared, response_train)

predictions= voting_class.predict(test_set_prepared)

tp, fn, fp, tn = confusion_matrix(response_test, predictions, labels=["Yes", "No"]).ravel()

precision = tp/(tp+fp)
precision
recall = tp/(tp+fn)
recall



################
## CLUSTERING ##
################

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(train_set_prepared)

kmeans.inertia_

kmeans.labels_


nb_clusters = np.arange(1,11,1)
list_inertia = []

for nb in nb_clusters:
    kmeans = KMeans(n_clusters = nb)
    kmeans.fit(train_set_prepared)
    list_inertia.append(kmeans.inertia_)


import matplotlib.pyplot as plt

plt.plot(nb_clusters,list_inertia)


final_kmeans = KMeans(n_clusters = 4)
final_kmeans.fit(train_set_prepared)


final_kmeans.labels_

train_set['cluster'] = final_kmeans.labels_


train_set[train_set['cluster']==0]
train_set[train_set['cluster']==1]
train_set[train_set['cluster']==2]
train_set[train_set['cluster']==3]


