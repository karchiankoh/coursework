'''
PROJECT DESCRIPTION:
Use the EM algorithm to do image segmentation. For this project, you are
required to segment the foreground from the background, i.e. the number of segments 𝐾 =
2. We use image colors in CIE-Lab color space as the observed data 𝑥J ∈ ℝn for the image
segmentation task. The Lab color space describes mathematically all perceivable colors in
the three dimensions L for lightness and a and b for the color opponents green–red and
blue–yellow. We provide you the Lab color for each
pixel as the observed data.
'''

import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
from io_data import read_data, write_data
from scipy.stats import multivariate_normal

def data_to_arr(data):
	# Convert data into a format that is better for manipulation
	rows = int(data[-1, 0] + 1)
	cols = int(data[-1, 1] + 1)
	data_arr = data[:,2:]
	return data_arr

def initialise_parameters(data_arr, N, K, dim):
	# Initialise the means, covariances and mixing coefficients
	np.random.seed()
	mean = np.empty((K, dim))
	covariance = np.empty((K, dim, dim))
	mixing_coefficient = np.ones((K)) / K
	for k in range(K):
		mean[k] = data_arr[np.random.randint(0, N)]
		covariance[k] = np.cov(data_arr, rowvar=False)
	return mean, covariance, mixing_coefficient

def evaluate_log_likelihood(N, K, data_arr, mean, covariance, mixing_coefficient):
	# Evaluate the log likelihood
	log_likelihood = np.empty((N))
	for n in range(N):
		log_term = 0
		for k in range(K):
			log_term += mixing_coefficient[k] * multivariate_normal.pdf(data_arr[n], mean=mean[k], cov=covariance[k])
		log_likelihood[n] = np.log(log_term)
	return np.sum(log_likelihood)

def evaluate_responsibilities(N, K, data_arr, mean, covariance, mixing_coefficient):
	# Expectation step: Evaluate responsibilities using the current parameter values
	responsibilities = np.empty((len(data_arr), K))
	for n in range(N):
		numerator = np.empty(K)
		for k in range(K):
			numerator[k] = mixing_coefficient[k] * multivariate_normal.pdf(data_arr[n], mean=mean[k], cov=covariance[k])
		responsibilities[n] = numerator / np.sum(numerator)
	return responsibilities

def reestimate_parameters(K, dim, data_arr, responsibilities):
	# Maximisation step: Re-estimate parameters using the current responsibilities
	new_mean = np.empty((K, dim))
	new_covariance = np.empty((K, dim, dim))
	new_mixing_coefficient = np.empty((K))
	for k in range(K):
		k_responsibilities = responsibilities[:,k]
		new_mean[k] = np.average(data_arr, axis=0, weights=k_responsibilities)
		dist = data_arr - new_mean[k]
		new_covariance[k] = np.cov(dist, rowvar=False, aweights=k_responsibilities)
		new_mixing_coefficient[k] = np.average(k_responsibilities)
	return new_mean, new_covariance, new_mixing_coefficient

def determine_responsibilities(data_arr, max_iterations):
	# Determine responsibilities against a maximum of 10 iterations of Expectation-Maximisation
	# or until log likelihood or parameters converge
	K = 2
	N, dim = data_arr.shape
	mean, covariance, mixing_coefficient = initialise_parameters(data_arr, N, K, dim) # Initialise parameters
	log_likelihood = evaluate_log_likelihood(N, K, data_arr, mean, covariance, mixing_coefficient) # Evaluate the initial value of the log likelihood
	converge = False
	T = 0
	while not converge:
		T+=1
		responsibilities = evaluate_responsibilities(N, K, data_arr, mean, covariance, mixing_coefficient) # Expectation step
		new_mean, new_covariance, new_mixing_coefficient = reestimate_parameters(K, dim, data_arr, responsibilities) # Maximisation step
		new_log_likelihood = evaluate_log_likelihood(N, K, data_arr, new_mean, new_covariance, new_mixing_coefficient) # Evaluate log likelihood
		if T==max_iterations or np.allclose(log_likelihood, new_log_likelihood) or ( np.allclose(mean, new_mean) and np.allclose(covariance, new_covariance) and np.allclose(mixing_coefficient, new_mixing_coefficient) ):
			break # Break if parameters or log likelihood converges or max iterations is reached
		else:
			mean = new_mean
			covariance = new_covariance
			mixing_coefficient = new_mixing_coefficient
			log_likelihood = new_log_likelihood
	return responsibilities

def segment_image(data, responsibilities):
	# Segment image based on responsibilities computed
	black_data = np.copy(data)
	black_data[:,2:5] = 0
	white_data = np.copy(data)
	white_data[:,2:5] = [100, 0, 0]
	mask_list = []
	seg1_list = []
	seg2_list = []
	for i in range(len(data)):
		if responsibilities[i][0] > responsibilities[i][1]:
			mask_list.append(black_data[i])
			seg1_list.append(data[i])
			seg2_list.append(black_data[i])
		else:
			mask_list.append(white_data[i])
			seg1_list.append(black_data[i])
			seg2_list.append(data[i])
	return np.asarray(mask_list).astype(np.float32), np.asarray(seg1_list).astype(np.float32), np.asarray(seg2_list).astype(np.float32)

def write_read_data(start_name, end_name, data):
	# save data matrix and image into text and jpg files respectively
	txt = "../output/" + start_name + "_" + end_name + ".txt"
	write_data(data, txt)
	jpg = "../output/" + start_name + "_" + end_name + ".jpg"
	read_data(txt, False, False, True, jpg)

def main():
	# Segment cow, fox, owl and zebra images using Expectation-Maximisation 
	# and save corresponding mask, seg1 and seg2 into text and jpg files
	names = ['cow', 'fox', 'owl', 'zebra']
	for a in range(len(names)):
		inputtxt = "../a2/" + names[a] + ".txt"
		print(inputtxt)
		data, image = read_data(inputtxt, False)
		data_arr = data_to_arr(data)
		responsibilities = determine_responsibilities(data_arr, 10)
		mask_data, seg1_data, seg2_data = segment_image(data, responsibilities)
		write_read_data(names[a], "mask", mask_data)
		write_read_data(names[a], "seg1", seg1_data)
		write_read_data(names[a], "seg2", seg2_data)

main()