'''
PROJECT DESCRIPTION:
Use the Gibbs sampling algorithm to denoise the
corrupted input data (binary images) we provide.
The input data is a binary image corrupted with noise. Two types of data files are
provided, you may choose to read from either file as the input data. The first data file is in
the PNG format (an image file format). The second data file is in the text format, where
each line gives you a coordinate and the corresponding binary value denoted as {0,255}.
The output that you will generate is a binary image with the noise removed. The
output file can be either in PNG or text format.
'''

import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
from io_data import read_data, write_data
import copy
import math
from scipy.stats import norm

def data_to_arr(data):
	# Convert data into a format that is better for manipulation
	rows = int(data[-1, 0] + 1)
	cols = int(data[-1, 1] + 1)
	data_arr = np.ones((rows, cols))
	for d in data:
		if d[2]==0:
			data_arr[int(d[0])][int(d[1])] = -1
	return data_arr

def arr_to_data(data_arr):
	# Convert data into the relevant format for writing into a text file
	data = []
	for i in range(data_arr.shape[0]):
		for j in range(data_arr.shape[1]):
			if data_arr[i][j]==-1:
				aList = [i, j, 0]
			else:
				aList = [i, j, 255]
			data.append(aList)
	data = np.asarray(data).astype(np.float32)
	return data

def compute_neighbour_compatibility(x, y, posterior_arr, posterior_value):
	# Determine neighbour compatibility by examining pairwise potential
	coupling_strength = 1
	num_agree = 0
	num_disagree = 0
	for i in range(x-1,x+2,2):
		for j in range(y-1,y+2,2):
			if i<posterior_arr.shape[0] and j<posterior_arr.shape[1]:
				if posterior_arr[i,j]==posterior_value:
					num_agree += 1
				else:
					num_disagree += 1
	neighbour_compatibility = 2 * coupling_strength * posterior_value * (num_agree - num_disagree)
	return neighbour_compatibility

def refresh_posterior(x, y, posterior_arr, posterior_value, log_evidence_term):
	# Refresh posterior of each pixel determined by compatiblity with its neighbours 
	# and compatibility with the data
	neighbour_compatibility = compute_neighbour_compatibility(x, y, posterior_arr, posterior_value)
	p = 1 / ( 1 + math.exp(-neighbour_compatibility + log_evidence_term) )
	if np.random.binomial(1, p) == 0:
		posterior_arr[x,y] = -1
	else:
		posterior_arr[x,y] = 1

def gibbs_sample(data_arr, iterations):
	# Run gibbs sampling against the number of iterations by computing the log of the
	# evidence terms. Then, referesh the posterior by the number of times specified in iterations
	new_data_arr = copy.deepcopy(data_arr)
	sigma = np.std(data_arr)
	log_evidence_term = np.log( norm.pdf(-1, data_arr, sigma) / norm.pdf(1, data_arr, sigma))
	for i in range(iterations):
		for x in range(data_arr.shape[0]):
			for y in range(data_arr.shape[1]):
				refresh_posterior(x,y, new_data_arr, new_data_arr[x,y], log_evidence_term[x,y])
	return new_data_arr

def main():
	# Denoise noise images 1-4 using the gibbs sampling algorithm and save the
	# denoised data matrix and denoised image into text and png files respectively
	for a in range(1,5):
		noisetxt = "../a1/" + str(a) + "_noise.txt"
		print(noisetxt)
		data, image = read_data(noisetxt, True)
		data_arr = data_to_arr(data)
		new_data_arr = gibbs_sample(data_arr, 15)
		new_data = arr_to_data(new_data_arr)
		denoisetxt = "../output/" + str(a) + "_denoise_gibbs.txt"
		write_data(new_data, denoisetxt)
		denoisepng = "../output/" + str(a) + "_denoise_gibbs.png"
		read_data(denoisetxt, True, False, True, denoisepng)

main()