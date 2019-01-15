'''
ASSIGNMENT DESCRIPTION:
You are required to implement the function “fit(self, X, y) ” and “predict(self, X)” in the file “DecisionTreeRegressor.py”.
In the function “fit(self, X, y) ”, you should update the self.node , whose type is dictionary. There are four keys in self.node, which are “splitting_variable ”, “splitting_threshold ”, “left ” and “right ”. The value of “splitting_variable” should be an integer number. The value of “splitting_threshold ” should be a float number.
The values of “left ” and “right ” should be either a float number or a dictionary.
'''

import numpy as np
import os
import json
import operator

class MyDecisionTreeRegressor():
	def __init__(self, max_depth=5, min_samples_split=1):
		'''
		Initialization
		:param max_depth: type: integer
		maximum depth of the regression tree. The maximum
		depth limits the number of nodes in the tree. Tune this parameter
		for best performance; the best value depends on the interaction
		of the input variables.
		:param min_samples_split: type: integer
		minimum number of samples required to split an internal node:

		root: type: dictionary, the root node of the regression tree.
		'''
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.root = None

	def select_splitting_variable_threshold(self, X, y):
		# Traverse the splitting variable 𝑗 and scan the splitting threshold 𝑠 for the fixed splitting variable 𝑗, and determine the pair (𝑗,𝑠) that minimizes the expression
		for j in range(len(X[0])): 
			allS = np.unique(X[:,j])

			for sIndex in range(len(allS)-1): 
				s = allS[sIndex]
				Y1 = []
				Y2 = []

				for index in range(len(X)):
					if X[index][j] <= s:
						Y1.append(y[index])
					else:
						Y2.append(y[index])

				c1 = np.average(Y1)
				c2 = np.average(Y2)
				criteria1 = (Y1-c1)**2
				criteria2 = (Y2-c2)**2
				sum_criteria = criteria1.sum() + criteria2.sum()

				if j==0 and sIndex==0:
					splitting_variable = j
					splitting_threshold = s
					min_sum = sum_criteria
				elif sum_criteria < min_sum:
					splitting_variable = j
					splitting_threshold = s
					min_sum = sum_criteria

		return splitting_variable, splitting_threshold

	def split_region_and_output(self, X, y, splitting_variable, splitting_threshold):
		# Split the region with the selected pair (𝑗, 𝑠) and determine the corresponding output value
		left_X_arr = []
		right_X_arr = []
		left_y_arr = []
		right_y_arr = []

		for index in range(len(X)):
			xRecord = X[index]
			if xRecord[splitting_variable] <= splitting_threshold:
				left_X_arr.append(xRecord)
				left_y_arr.append(y[index])
			else:
				right_X_arr.append(xRecord)
				right_y_arr.append(y[index])

		left_X = np.array(left_X_arr)
		right_X = np.array(right_X_arr)
		left_y = np.array(left_y_arr)
		right_y = np.array(right_y_arr)
		left_c = np.average(left_y_arr)
		right_c = np.average(right_y_arr)

		return left_X, right_X, left_y, right_y, left_c, right_c

	def construct_decision_tree(self, X, y, depth):
		# Select the optimal splitting variable 𝑗 and the splitting threshold 𝑠
		splitting_variable, splitting_threshold = self.select_splitting_variable_threshold(X, y)

		model_dict = {
			"splitting_variable": splitting_variable,
			"splitting_threshold": splitting_threshold
		}

		# Split the region with the selected pair (𝑗, 𝑠) and determine the corresponding output value
		left_X, right_X, left_y, right_y, left_c, right_c = self.split_region_and_output(X, y, splitting_variable, splitting_threshold)

		# Repeat the above two steps considering each resulting region as a parent node until the maximum depth of the tree is obtained
		if len(left_X)<self.min_samples_split or depth==1:
			model_dict["left"] = left_c
		else:
			model_dict["left"] = self.construct_decision_tree(left_X, left_y, depth-1)
		if len(right_X)<self.min_samples_split or depth==1:
			model_dict["right"] = right_c
		else:
			model_dict["right"] = self.construct_decision_tree(right_X, right_y, depth-1)

		return model_dict

	def fit(self, X, y):
		'''
		Inputs:
		X: Train feature data, type: numpy array, shape: (N, num_feature)
		Y: Train label data, type: numpy array, shape: (N,)

		You should update the self.root in this function.
		'''
		# The input space is divided into M regions R1, 𝑅2, … , 𝑅𝑀, and combined to form the decision tree
		self.root = self.construct_decision_tree(X, y, self.max_depth)

	def predict_x(self, model_dict, xRecord):
		splitting_variable = model_dict["splitting_variable"]
		splitting_threshold = model_dict["splitting_threshold"]

		if xRecord[splitting_variable] <= splitting_threshold:
			nextNode = model_dict["left"]
		else:
			nextNode = model_dict["right"]
		if type(nextNode) is dict:
			return self.predict_x(nextNode, xRecord)
		else:
			return nextNode
		
	def predict(self, X):
		'''
		:param X: Feature data, type: numpy array, shape: (N, num_feature)
		:return: y_pred: Predicted label, type: numpy array, shape: (N,)
		'''
		y_pred = []

		for xRecord in X:
			y = self.predict_x(self.root, xRecord)
			y_pred.append(y)

		return np.array(y_pred)

	def get_model_string(self):
		model_dict = self.root
		return model_dict

	def save_model_to_json(self, file_name):
		model_dict = self.root
		with open(file_name, 'w') as fp:
			json.dump(model_dict, fp)


# For test
if __name__=='__main__':
	for i in range(3):
		x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
		y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")
		
		for j in range(2):
			tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
			tree.fit(x_train, y_train)

			model_string = tree.get_model_string()

			with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
				test_model_string = json.load(fp)

			print(operator.eq(model_string, test_model_string))

			y_pred = tree.predict(x_train)

			y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
			print(np.square(y_pred - y_test_pred).mean() <= 10**-10)
