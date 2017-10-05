import os
import sys

"""
     USAGE : python mktruth.py <path to train_folder>  <truth_file_name>


"""

join = lambda u,v: os.path.join(u,v)
isfile = lambda z : os.path.isfile(z)
isdir = lambda z : os.path.isdir(z)

if __name__ == '__main__':

	if isdir(sys.argv[1]):
		tp = sys.argv[1]
	else:
		print 'invalid directory'

	truth_file_name = sys.argv[2]

	direcs = [ d for d in os.listdir(tp) if isdir(join(tp,d))]
	#categories = [x for x in xrange(len(direcs))]

	with open('./'+truth_file_name+'.txt','w') as out:
		
		#for direc,categ in zip(direc,categories):
		for direc in direcs:
			fnames = [ f for f in os.listdir(join(tp,direc)) if isfile(join(join(tp,direc),f))]
			for fn in fnames:
				out.write('{0},{1}'.format(fn,direc))
				#out.write(fn,categ)
				out.write('\n')



