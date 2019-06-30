import os
import io
import sys
import numpy as np
import pandas as pd
from os import walk
from sklearn.metrics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
# plt.style.use('seaborn-white')
from sklearn import svm

'''
################################################################################################
Data Preprocessing and feature extraction
################################################################################################
'''

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

folder = "TrainData"
websites = []
for(_,_,filenames) in walk(folder):
	websites.extend(filenames)

print('-- Websites count',len(websites))

features = []
labels = []

print('-- Labels distribution',labels.count(0),labels.count(1))

''' for normalization '''
# sx = StandardScaler().fit(features)
# x = sx.transform(features)


ff = pd.DataFrame(pd.read_csv(folder + "/" + "website_data_1.csv"))
fx = list(ff.dtypes.index)
fx.pop(0)
x = np.array(features)
y = np.array(labels)


''' OUTLIER DETECTION ISOLATION FOREST '''

print "outlier detection"
clf = IsolationForest(contamination=0.1)
clf.fit(x)
out_res = clf.predict(x)
#print zip(out_res,labels)
print "Total points : ",len(out_res)
print "Inliners : ",np.count_nonzero(out_res == 1)
print "Outliers : ",np.count_nonzero(out_res == -1)

nx = []
ny = []
for i in xrange(len(out_res)):
	if out_res[i] == 1:
		nx.append(x[i])
		ny.append(y[i])

print('-- IF => Labels distribution',ny.count(0),ny.count(1))

x = np.array(nx)
y = np.array(ny) 

'''
################################################################################################
Training and Model Analysis
################################################################################################
'''

model = [RandomForestClassifier(n_estimators = 25),svm.SVC(),KNeighborsClassifier(),
	MLPClassifier(hidden_layer_sizes=(3,),shuffle=True,random_state=1, max_iter=20, warm_start=True),GaussianNB()]
m = ["RandomForest","SVM","KNN","MLP","NB"]

for i in range(len(model)):
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
	model[i].fit(x_train, y_train)
	pred=model[i].predict(x_test)
	print m[i],"accuracy : ",accuracy_score(y_test,pred)
	cm = confusion_matrix(y_test,pred)
	cr = classification_report(y_test,pred)

	#imp = model[i].feature_importances_

''' to draw barchart '''
sn.heatmap(cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("cm.png")

''' to draw barchart '''
dk = zip(fx,imp)
dz = sorted(dk,key=lambda x:(-x[1],x[0]))
print dz
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(dz)), [val[1] for val in dz], align='center')
plt.xticks(range(len(dz)), [val[0] for val in dz])
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("imp4.png")

''' add in the for loop  above for k fold cross validation '''
	# kf = KFold(n_splits=10,shuffle=True)
	# mean_acc = 0
	# for train_index, test_index in kf.split(x):
	# 	x_train, x_test = x[train_index], x[test_index]
	# 	y_train, y_test = y[train_index], y[test_index]
	# 	model[i].fit(x_train, y_train)
	# 	pred=model[i].predict(x_test)
	# 	mean_acc += accuracy_score(y_test,pred)
	# print m[i],"cross_val accuracy : ",mean_acc/10


'''
################################################################################################
TESTING
################################################################################################
 '''

testFeatures = []

# use try and except to count the number of error instances

testPredictions = model[0].predict(testFeatures)

#probability of the predicetd label
prob = model[0].predict_proba(testFeatures)
