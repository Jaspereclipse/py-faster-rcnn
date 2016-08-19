# Visualize the losses in the log file
# Written by: Juanyan Li

import re
import sys, os
import argparse
from collections import defaultdict as dd
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

def parse_args():

	parser = argparse.ArgumentParser(description='Visualize loss function from log file')
	parser.add_argument("--log", dest="log_file", help="log file directory",
		default=None, type=str)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args

def parse_patterns(ls, patterns, rule=None, data_type=float):
	dict_ = dd(list)
	len_check = None
	
	if rule is None:
		rule = (lambda pattern: "(?<= " + pattern + " = )" + "\d*.?\d*")
	
	for pattern in patterns:
		regex = rule(pattern)
		dict_[pattern] = np.array([re.search(regex, line).group(0) for line in content if re.search(regex, line) is not None], dtype=data_type)

		# make sure we have the same amount of losses for each pattern
		if len_check is None:
			len_check = len(dict_[pattern])
		else:
			assert len_check == len(dict_[pattern])
	return dict_

def plot_losses(losses, iteration, colors, select=None, use_conv=False):
	legend = losses.keys()
	if select is not None and len(select) > 0:
		legend =  list(set(select).intersection(legend))
		assert len(legend) > 0
	num_losses = len(legend)
	
	# Set up colors
	assert len(colors) == num_losses
	fig = plt.figure()
	plt.rc('axes', prop_cycle=(cycler('color', colors)))
	
	# Customize
	# axes = plt.gca()
	# axes.set_ylim([0,3])

	if use_conv:
		N = 100 # average over 100 points
		conv_avg = (lambda l, kernel=np.ones((N,))/N: np.convolve(l, kernel, mode='valid'))

	# Plot each loss
	for l in legend:
		if use_conv:
			loss_conv = conv_avg(losses[l])
			iter_conv = iteration[-len(loss_conv):]
			plt.plot(iter_conv, loss_conv)
		else:
			plt.plot(iteration, losses[l])
	
	# Details
	plt.legend(legend, loc='upper right')
	plt.xlabel('Iterations')
	plt.ylabel('Loss')


	# Show it
	plt.show()




if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)

	# Sanity check
	assert os.path.isfile(args.log_file)

	with open(args.log_file, 'r') as f:
		content = f.readlines()

	# parse the losses lines
	patterns = ["loss", "loss_bbox", "loss_cls", "rpn_cls_loss", "rpn_loss_bbox"]
	iter_pattern = ["Iteration"]
	iter_rule = (lambda p: "(?<= " + p + " )" + "\d*")
	losses = parse_patterns(content, patterns)
	total_iter = parse_patterns(content, iter_pattern, rule=iter_rule, data_type=int)
	total_iter = total_iter[iter_pattern[0]][::2]
	assert len(total_iter) == len(losses[patterns[0]])

	# plot the losses separately and in one graph
	plot_losses(losses, total_iter, colors=['r'], select=["loss"], use_conv=True)
	plot_losses(losses, total_iter, colors=['g'], select=["loss_bbox"], use_conv=True)
	plot_losses(losses, total_iter, colors=['b'], select=["loss_cls"], use_conv=True)
	plot_losses(losses, total_iter, colors=['y'], select=["rpn_cls_loss"], use_conv=True)
	plot_losses(losses, total_iter, colors=['k'], select=["rpn_loss_bbox"], use_conv=True)
	plot_losses(losses, total_iter, colors=['r', 'g', 'b', 'y', 'k'], use_conv=True)