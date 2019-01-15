'''
PROJECT DESCRIPTION:
Use variational inference algorithm to denoise the
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
			if round(data_arr[i][j])==-1:
				aList = [i, j, 0]
			else:
				aList = [i, j, 255]
			data.append(aList)
	data = np.asarray(data).astype(np.float32)
	return data

def compute_mean_field_influence(x, y, mu_arr):
	# Compute mean field influence of each pixel by summing the product of coupling strength 
	# and the average values of each neighbour
	coupling_strength = 1
	neighbours_sum = 0
	for i in range(x-1,x+2,2):
		for j in range(y-1,y+2,2):
			if i<mu_arr.shape[0] and j<mu_arr.shape[1]:
				neighbours_sum += mu_arr[i,j]
	mean_field_influence = coupling_strength * neighbours_sum
	return mean_field_influence

def compute_mu(data_arr, iterations):
	# Evaluate mu by computing the difference between the strengths of L. Then, compute the  
	# mean field influence and subsequently, mu until it converges or by the maximum number of 
	# times specified by iterations
	sigma = np.std(data_arr)
	mu_arr = copy.deepcopy(data_arr)
	l_difference = norm.logpdf(data_arr, 1, sigma) - norm.logpdf(data_arr, -1, sigma)
	for i in range(iterations):
		mu_temp_arr = copy.deepcopy(mu_arr)
		for x in range(data_arr.shape[0]):
			for y in range(data_arr.shape[1]):
				mean_field_influence = compute_mean_field_influence(x, y, mu_arr)
				mu_temp_arr[x,y] = math.tanh(mean_field_influence + 0.5 * l_difference[x,y])
		if np.allclose(mu_arr, mu_temp_arr):
			break
		mu_arr = mu_temp_arr
	return mu_arr

def main():
	# Denoise noise images 1-4 using the gibbs sampling algorithm and save the
	# denoised data matrix and denoised image into text and png files respectively
	for a in range(1,5):
		noisetxt = "../a1/" + str(a) + "_noise.txt"
		print(noisetxt)
		data, image = read_data(noisetxt, True)
		data_arr = data_to_arr(data)
		mu_arr = compute_mu(data_arr, 15)
		mu_data = arr_to_data(mu_arr)
		denoisetxt = "../output/" + str(a) + "_denoise_variational.txt"
		write_data(mu_data, denoisetxt)
		denoisepng = "../output/" + str(a) + "_denoise_variational.png"
		read_data(denoisetxt, True, False, True, denoisepng)

main()