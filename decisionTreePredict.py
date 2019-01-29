from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
#import graphviz


def get_data():
    if os.path.exists("data.csv"):
        print("-- data.csv found locally")
        df = pd.read_csv("data.csv", index_col=0)
    else:
        print("data not found")
        
    return df


#print("df.head() ", df.head(), sep="\n", end="\n\n")

#print("* Diagnosis types:", df["diagnosis"].unique(), sep="\n")


def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name : n for n, name in enumerate(targets)}
    df_mod["diagnosis"] = df_mod[target_column].replace(map_to_int)
    
    return(df_mod, targets)


'''
print("* df2.head()", df2[["Target", "diagnosis"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df2[["Target", "diagnosis"]].tail(),
      sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")
'''


#print("features;",features,sep="\n")


def visualize_tree(tree, features_names, filename):
    with open(filename, 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=features_names,
                        class_names=class_names )
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")
        '''
df = get_data()
features = list(df.columns[1:31])
df, targets = encode_target(df, "diagnosis")        
df2, targets = encode_target(df, "diagnosis")

y = df2["diagnosis"]

X = df2[features]
print(X)
#print(X)
#X = df["compactness_mean"]

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99, 
                            criterion="gini")
dt.fit(X, y)

visualize_tree(dt, features)
    '''
filename = "giniTrainTest.dot" #must be  .dot file
filename_en = "entropyTrainTest.dot"
filename_full= "giniFull.dot"    
filename_en_full = "entropyFull.dot"
visualizeTree = False
    
#Import data
df = get_data()
print("Dataset Lenght:: ", len(df))
print("Dataset shpae::", df.shape)

class_names = ["Malign", "Benign"]
#Features and encode Bening and Malign into Ints
features = list(df.columns[1:31])
df, targets = encode_target(df, "diagnosis")
X = df.values[:,1:31]
Y = df.values[:,0]

#split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.3, random_state=100)
#Gini Train 
clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
clf_gini.fit(X_train, y_train)
if(visualizeTree):
    visualize_tree(clf_gini, features, filename)
#Gini Full
clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
clf_gini.fit(X, Y)
if(visualizeTree):
    visualize_tree(clf_gini, features, filename_full)
    
#Entropy Train
clf_entropy = DecisionTreeClassifier(criterion = "entropy",  random_state = 100,
                                     max_depth=2, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
if(visualizeTree):
    visualize_tree(clf_entropy, features, filename_en)
#Entropy Full
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=2, min_samples_leaf=5)
clf_entropy.fit(X, Y)
if(visualizeTree):
    visualize_tree(clf_entropy, features, filename_en_full)

y_pred = clf_gini.predict(X_test)
#print(y_pred)

y_pred_en = clf_entropy.predict(X_test)

y_pred_Full_gini = clf_gini.predict(X)
y_pred_Full_Entr = clf_entropy.predict(X)

#Accuracy for Gini
print("Accuracy of gini train in test is ", accuracy_score(y_test, y_pred)*100)
#Accuracy for Entropy
print("Accuracy of Entropy train in test is ", accuracy_score(y_test, y_pred_en)*100)
#Accuracy for Gini Full dataset
print("Accuracy of gini Full dataset in test is ", accuracy_score(Y, y_pred_Full_gini)*100)
#Accuracy for Entropy Full dataset
print("Accuracy of Entropy Full dataset in test is ", accuracy_score(Y, y_pred_Full_Entr)*100)
#Positives and Negatives
tp=fp=fn=tn=0
for x in range(len(y_pred_Full_Entr)):
    if(Y[ x] == 1) :
        if(y_pred_Full_Entr[x]==1):
            tp+=1
        elif(y_pred_Full_Entr[x]==0):
            fp+=1
    elif (Y[x] < y_pred_Full_Entr[x]):
        fn +=1
    elif(Y[ x] == 0 ):
        if(y_pred_Full_Entr[x]==0):
            tn +=1


sensitivity_Full_Entr=  tp/(tp+fn)
specificity_Full_Entr = tn/(tn+fp)

tp=fp=fn=tn=0
for x in range(len(y_pred_Full_gini)):
    if(Y[ x] == 1) :
        if(y_pred_Full_gini[x]==1):
            tp+=1
        elif(y_pred_Full_gini[x]==0):
            fp+=1
    elif (Y[x] < y_pred_Full_gini[x]):
        fn +=1
    elif(Y[ x] == 0 ):
        if(y_pred_Full_gini[x]==0):
            tn +=1


sensitivity_Full_Gini=  tp/(tp+fn)
specificity_Full_Gini = tn/(tn+fp)

tp=fp=fn=tn=0
for x in range(len(y_pred)):
    if(Y[ x] == 1) :
        if(y_pred[x]==1):
            tp+=1
        elif(y_pred[x]==0):
            fp+=1
    elif (Y[x] < y_pred[x]):
        fn +=1
    elif(Y[ x] == 0 ):
        if(y_pred[x]==0):
            tn +=1


sensitivity_Gini=  tp/(tp+fn)
specificity_Gini = tn/(tn+fp)

tp=fp=fn=tn=0
for x in range(len(y_pred_en)):
    if(Y[ x] == 1) :
        if(y_pred_en[x]==1):
            tp+=1
        elif(y_pred_en[x]==0):
            fp+=1
    elif (Y[x] < y_pred_en[x]):
        fn +=1
    elif(Y[ x] == 0 ):
        if(y_pred_en[x]==0):
            tn +=1


sensitivity_Entr=  tp/(tp+fn)
specificity_Entr = tn/(tn+fp)

#Sensitivity and Specificity for Gini
print("Sensitivity using Gini ", sensitivity_Gini)
print("Specificity using Gini ", specificity_Gini)
#Sensitivity and Specificity for Entropy
print("Sensitivity using Entropy ", sensitivity_Entr)
print("Specificity using Entropy ", specificity_Entr)
#Sensitivity and Specificity for Gini Full dataset
print("Sensitivity using Gini Full dataset ", sensitivity_Full_Gini)
print("Specificity using Gini Full dataset ", specificity_Full_Gini)
#Sensitivity and Specificity for Gini Full dataset
print("Sensitivity using Entropy Full dataset ", sensitivity_Full_Entr)
print("Specificity using Entropy Full dataset ", specificity_Full_Entr)




#clf_gini.predict_proba(X_test[:1, :])
'''

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
dot_data = export_graphviz(clf)
'''



