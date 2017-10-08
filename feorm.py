import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,classification_report,make_scorer
from sklearn.ensemble import RandomForestClassifier as RFC,AdaBoostClassifier as Ada
from sklearn.model_selection import cross_validate
import argparse

"""
     USAGE : python feorm.py --train <train-csv> --test <test-csv> --fnames <unpruned-test-arff file>

     outputs submission.txt
"""

relabel = lambda x: 0 if x=='female'  else 1
unlabel = lambda y: 'female' if y==0  else 'male' 

def get_data_frame(csv_file,istrain=True):
	df = pd.read_csv(csv_file)
	# for now let this be as you are on dev set 
	#if istrain:
	labels = np.asarray([ relabel(row[-1]) for row in df.values])
	df = df.drop(labels=df.columns[-1],axis=1)
	return df,labels

def get_fnames(unpruned_arff):
	fnames = []
	with open(unpruned_arff) as af:
		data_flag = False
		for line in af:
			pieces = line.split()
			if data_flag:
				fnames.append(pieces[-1].replace('_','.'))

			if pieces[0] == '@data':
				data_flag = True
	return fnames


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='python feorm.py --train <train-csv> --test <test-csv> --fnames <unpruned-test-arff file>')
	parser.add_argument('-tr','--train',help='path to arff file',required=True)
	parser.add_argument('-te','--test',help='path to arff file',required=True)
	parser.add_argument('-fn','--fnames',help='path to arff file',required=True)
	args= vars(parser.parse_args())

	trdf , labels = get_data_frame(args['train'])
	#tedf , truth = get_data_frame(args['test'])
	#fnames = get_fnames(args['fnames'])

	"""#### for pyramidal
	clf = RFC(n_estimators=50,
	 criterion='entropy', 
	 max_depth=None, 
	 min_samples_split=2, 
	  max_features=10, 
	  max_leaf_nodes=None,
	  class_weight = 'balanced'
	)
	#########"""
	clf1 = svm.SVC(C=1,kernel='poly',gamma = 'auto')
	clf2 = svm.SVC(C=1,kernel='rbf',gamma = 'auto')
	clf3 = svm.SVC(C=1,kernel='linear',gamma = 'auto')
	svms = (clf1,clf2,clf3)
	
	scoring = ['f1_macro','accuracy','precision_macro','recall_macro']
	for clf in svms:
		scores = cross_validate(clf,trdf,labels,scoring = scoring,cv=10,return_train_score=False)
		for s in scores.keys():
			print s
			for v in scores[s]:
				print v
			print '###############################'
		print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
		

	#clf.fit(trdf,labels)
	"""clf1.fit(trdf,labels)
	clf2.fit(trdf,labels)
	clf3.fit(trdf,labels)"""

	#z = clf.predict(tedf)
	"""z1 = clf1.predict(tedf)
	z2 = clf2.predict(tedf)
	z3 = clf3.predict(tedf)"""
	
	#print 'RF : '
	#print '{0}'.format(classification_report(y_true=truth,y_pred=z,target_names=['female','male']))
	#print 'SVC linear: '
	#print '{0}'.format(classification_report(y_true=truth,y_pred=z3,target_names=['female','male']))


