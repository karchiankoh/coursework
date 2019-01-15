'''
ASSIGNMENT DESCRIPTION:
You are required to implement the function “fit(self, X, y) ” and “predict(self, X)” in the file “GradientBoostingRegressor.py”.
In the function “fit(self, X, y) ”, you should update the self.estimators, whose type is np array. Each element in this array is a regression tree object.
'''

import numpy as np
from DecisionTreeRegressor import MyDecisionTreeRegressor
import os
import json
import operator
import copy

class MyGradientBoostingRegressor():
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param learning_rate: type:float
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        int (default=100)
        :param n_estimators: type: integer
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        :param max_depth: type: integer
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node

        estimators: the regression estimators
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = np.empty((self.n_estimators,), dtype=np.object)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.initial_c = 0

    def initialise_function(self, y):
        self.initial_c = np.average(y)
        function = np.full((len(y)), self.initial_c)
        return function

    def compute_residuals(self, X, y, function):
        # For 𝑖 = 1,2, … , 𝑁, compute the residual
        residuals = y - function
        return y - function

    def fit_regression_tree(self, X, y):
        # Fit a regression tree to the targets γim resulting in terminal regions 𝑅𝑚𝑗 , 𝑗 =1,2, … , 𝐽𝑚
        tree = MyDecisionTreeRegressor(self.max_depth, self.min_samples_split)
        tree.fit(X, y)
        return tree

    def compute_cmj_dict(self, cmj_dict):
        cmj_left = cmj_dict["left"]
        cmj_right = cmj_dict["right"]

        if type(cmj_left) is dict:
            self.compute_cmj_dict(cmj_left)
        else:
            cmj_dict["left"] = cmj_left * self.learning_rate
        if type(cmj_right) is dict:
            self.compute_cmj_dict(cmj_right)
        else:
            cmj_dict["right"] = cmj_right * self.learning_rate

    def compute_cmjs(self, residual_dict):
        # For 𝑗 = 1,2, … , 𝐽𝑚, compute 𝑐𝑚𝑗
        cmj_dict = copy.deepcopy(residual_dict)
        self.compute_cmj_dict(cmj_dict)
        cmj_tree = MyDecisionTreeRegressor(self.max_depth, self.min_samples_split)
        cmj_tree.root = cmj_dict
        return cmj_tree

    def update_function(self, X, function, cmj_tree):
        pred = cmj_tree.predict(X)
        function = function + pred
        return function

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.estimators in this function
        '''
        # Initialize 𝑓0(𝑥)
        function = self.initialise_function(y)

        # For 𝑚 = 1 to 𝑀:
        for m in range(self.n_estimators):
            # For 𝑖 = 1,2, … , 𝑁, compute the residual
            residuals = self.compute_residuals(X, y, function)
            # Fit a regression tree to the targets γim resulting in terminal regions 𝑅𝑚𝑗 , 𝑗 =1,2, … , 𝐽𝑚
            residual_tree = self.fit_regression_tree(X, residuals)
            self.estimators[m] = residual_tree
            # For 𝑗 = 1,2, … , 𝐽𝑚, compute 𝑐𝑚𝑗
            cmj_tree = self.compute_cmjs(residual_tree.root)
            # Update fm(𝑥)
            function = self.update_function(X, function, cmj_tree)

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_pred = np.full((len(X)), self.initial_c)

        for m in range(self.n_estimators):
            residual_tree = self.estimators[m]
            residual_pred = residual_tree.predict(X)
            y_pred = y_pred + self.learning_rate * residual_pred

        return y_pred

    def get_model_string(self):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})

        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            n_estimators = 10 + j * 10
            gbr = MyGradientBoostingRegressor(n_estimators=n_estimators, max_depth=5, min_samples_split=2)
            gbr.fit(x_train, y_train)
            
            model_string = gbr.get_model_string()

            with open("Test_data" + os.sep + "gradient_boosting_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_string = json.load(fp)

            print(operator.eq(model_string, test_model_string))
            
            y_pred = gbr.predict(x_train)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_gradient_boosting_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            print(np.square(y_pred - y_test_pred).mean() <= 10**-10)
