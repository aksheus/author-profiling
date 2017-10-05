import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,classification_report
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
	tedf , truth = get_data_frame(args['test'])
	fnames = get_fnames(args['fnames'])

	clf1 = svm.SVC()
	clf2 = svm.NuSVC()
	clf3 = svm.LinearSVC()

	clf1.fit(trdf,labels)
	clf2.fit(trdf,labels)
	clf3.fit(trdf,labels)

	z1 = clf1.predict(tedf)
	z2 = clf2.predict(tedf)
	z3 = clf3.predict(tedf)

	print 'SVC : {0}'.format(classification_report(y_true=truth,y_pred=z1,target_names=['female','male']))
	print 'NuSVC : {0}'.format(classification_report(y_true=truth,y_pred=z2,target_names=['female','male']))
	print 'LinearSVC : {0}'.format(classification_report(y_true=truth,y_pred=z3,target_names=['female','male']))

