import os
import sys
from shutil import copy
from random import shuffle 
"""
 USAGE: python dev-move.py <source dir> <tatget-train> <target-test> <percent-for-train>

"""

join = lambda x,y: os.path.join(x,y)
isdir = lambda x: os.path.isdir(x)
isfile = lambda y: os.path.isfile(y)

if __name__ == '__main__':

	if isdir(sys.argv[1]):
		source = sys.argv[1]
	else:
		print 'invalid source dir'

	if isdir(sys.argv[2]):
		target_train = sys.argv[2]
	else:
		print 'invalid target train dir'

	if isdir(sys.argv[3]):
		target_test = sys.argv[3]
	else:
		print 'invalid target test dir'

	percentage = int(sys.argv[4])

	files  = [f for f in os.listdir(source) if isfile(join(source,f))]

	to_train = (len(files)*percentage) /100

	shuffle(files)

	for _ in xrange(to_train):
		f = files.pop()
		copy(join(source,f),join(target_train,f))

	while files:
		f=files.pop()
		copy(join(source,f),join(target_test,f))








