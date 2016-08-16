#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN Forked on ImageNet Dataset
# Written by Juanyan Li
# --------------------------------------------------------

"""Split the .txt file into train and test"""

import os, sys
import argparse
import numpy as np
from sklearn.cross_validation import ShuffleSplit

def parse_args():
	  """
	  Parse input arguments
	  """
	  parser = argparse.ArgumentParser(description='Shuffle and split data set for training and testing')
	  parser.add_argument('--des', dest='des', 
	  	help='.txt that specify the dataset (e.g. val.txt)',
	  	default=None, type=str)
	  parser.add_argument('--ts', dest='test_size',
	  	help='Size of test set (e.g. 0.25)', default=0.25, type=float)
	  parser.add_argument('--rnd', dest='random_state',
	  	help='Specify seed to reproducibility', default=None, type=int)

	  if len(sys.argv) == 1:
	  	parser.print_help()
	  	sys.exit(1)

	  args = parser.parse_args()
	  return args

if __name__ == '__main__':
		args = parse_args()

		print('Called with args:')
		print(args) 

		if not os.path.isfile(args.des) and args.des.endswith('.txt'):
			print("The text file specified is not valid.")
			sys.exit(1)

		with open(args.des, 'r') as f:
			content = f.readlines()
			content = np.array([c.split(" ")[0] for c in content])
		
		num_records = len(content)

		rs = ShuffleSplit(num_records, n_iter=1, test_size=args.test_size, 
			random_state=args.random_state)

		check = 0
		for train_index, test_index in rs:
			 print("TRAIN:", train_index, "TEST:", test_index)
			 train_set = content[train_index]
			 test_set = content[test_index]
			 check += 1

		assert check == 1

		train_index = range(1, 1 + len(train_set))
		test_index = range(1, 1 + len(test_set))
		train_set = [" ".join([t, str(i)]) for t, i in zip(train_set, train_index)]
		test_set = [" ".join([t, str(i)]) for t, i in zip(test_set, test_index)]

		output_dir = os.path.dirname(args.des)
		basename = os.path.split(args.des)[1].split('.')[0]
		train_name = "".join([basename, "_train", ".txt"])
		test_name = "".join([basename, "_test", ".txt"])
		train_name = os.path.join(output_dir, train_name)
		test_name = os.path.join(output_dir, test_name)

		print "Saving train set to: %s" %train_name
		print "Saving test set to: %s" %test_name


		np.savetxt(train_name, train_set, fmt='%s')
		np.savetxt(test_name, test_set, fmt='%s')