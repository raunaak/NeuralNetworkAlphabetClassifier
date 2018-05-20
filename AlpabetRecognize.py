
##												Digit Classifier - MNIST										 ##

## ----------------------------------------------------Imports-------------------------------------------------- ##
import csv
import time
import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from choose_alphabet import myOneHotEncoder
from choose_alphabet import chosen_digits
## ------------------------------------------------------------------------------------------------------------- ##


## ---------------------------------------Reading and preprocessing the data - csv------------------------------##

def preprocess():
	train_data = pd.read_csv('final_training_data.csv')
	test_data = pd.read_csv('final_test_data.csv')
	train_data = train_data.ix[:,:]
	images_training = np.asarray(train_data.ix[:,1:])
	images_training = images_training.T
	labels_training = np.asarray(train_data.ix[:,0])
	labels_training = labels_training.T
	labels_training = np.reshape(labels_training, (1, images_training.shape[1]))
	images_test = np.asarray(test_data.ix[:,1:])
	images_test = images_test.T
	labels_test = np.asarray(test_data.ix[:,0])
	labels_test = labels_test.T
	labels_test = np.reshape(labels_test, (1, images_test.shape[1]))
	labels_training = labels_training.reshape(labels_training.shape[1], 1)
	labels_training = myOneHotEncoder(labels_training)
	labels_training = labels_training.T
	labels_test = labels_test.reshape(labels_test.shape[1], 1)
	labels_test = myOneHotEncoder(labels_test)
	labels_test = labels_test.T

	images_training = images_training + 0 * np.random.normal(0, 1, images_training.shape)
	X_train = images_training
	Y_train = labels_training
	X_test = images_test
	Y_test = labels_test


	return X_train/255.0,Y_train,X_test/255.0,Y_test
## ------------------------------------------------------------------------------------------------------------- ##


## ---------------------------------------------initializing the parameters------------------------------------- ##

def initialize_weights(num_layers, node_list):
	Weights = {}
	for i in range(1, num_layers):
		w = np.random.randn(node_list[i], node_list[i-1])*0.1
		b = np.zeros((node_list[i],1))
		Weights['W'+str(i)] = w
		Weights['b'+str(i)] = b
		Weights['gamma'+str(i)] = np.random.randn(node_list[i], 1)
		Weights['beta'+str(i)] = np.random.randn(node_list[i], 1)
	return Weights

## ------------------------------------------------------------------------------------------------------------- ##

## ------------------------------------------activations and their derivatives---------------------------------- ##

def relu(z):
	z = np.maximum(z,0)
	return z

def softmax(x):
    return np.exp(x - x.max(axis=0))/np.sum(np.exp(x-x.max(axis=0)), axis = 0, keepdims=True)

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def activation_derivative(z, function):
	if (function == 'relu'):
		dz = z.copy()
		dz[z<=0] = 0
		dz[z>0] = 1
		return dz
	elif (function == 'sigmoid'):
		return np.multiply(sigmoid(z), 1-sigmoid(z))


## ------------------------------------------------forward propagation------------------------------------------ ##

def update_weights(Weights, dWeights, num_layers, learning_rate, regulariser, regularization_factor, batch_normalization = False):
	for i in range(1,num_layers):
		if (regulariser == 'L2'):
			Weights['W'+str(i)] = Weights['W'+str(i)] - learning_rate*dWeights['dW'+str(i)] - learning_rate * regularization_factor*Weights['W'+str(i)]
		elif(regulariser == 'L1'):
			Weights['W'+str(i)] = Weights['W'+str(i)] - learning_rate*dWeights['dW'+str(i)] - learning_rate * regularization_factor * np.sign(Weights['W'+str(i)])
		else:
			Weights['W' + str(i)] = Weights['W' + str(i)] - learning_rate * dWeights['dW' + str(i)]
		if batch_normalization:
			Weights["gamma"+str(i)] = Weights["gamma"+str(i)] - learning_rate*dWeights["dgamma"+str(i)]
			Weights["beta"+str(i)] = Weights["beta"+str(i)] - learning_rate*dWeights["dbeta"+str(i)]
		else:
			Weights['b'+str(i)] = Weights['b'+str(i)] - learning_rate * dWeights['db'+str(i)]
	return Weights

## ------------------------------------------------------------------------------------------------------------- ##

## ------------------------------------------Cross Entropy Cost & accuracy-------------------------------------- ##

def crosscost(string, Y_train, Y_predicted, regularizer, regularization_factor, Weights, num_layers):
	if string == 'Cross':
		cost = -np.sum(np.multiply(Y_train, np.log(Y_predicted)))
	elif string == 'MSE':
		cost = np.sum((Y_train - Y_predicted) * (Y_train - Y_predicted))
	if regularizer == 'L2':
		for i in range(1, num_layers):
			W = Weights['W' + str(i)]
			cost += regularization_factor * np.sum(W * W)/2
	elif regularizer == 'L1':
		for i in range(1, num_layers):
			W = Weights['W' + str(i)]
			cost += regularization_factor * np.sum(np.abs(W))
	return cost

def accuracyC(Y_train, Y_predicted):
	maxiesT = np.argmax(Y_train, axis=0)
	maxiesP = np.argmax(Y_predicted, axis = 0)
	ans = maxiesT.copy()
	ans[maxiesT == maxiesP] = 1
	ans[maxiesT != maxiesP] = 0
	return np.sum(ans)*1.0/Y_train.shape[1]

def total_feed_forward(Weights, X_train, sigmoids, num_layers, batch_normalization, drop_prob, testing=True):
	activations = {}
	drop_layer = np.random.binomial(1, drop_prob, X_train.shape)
	activations['D0'] = drop_layer
	activations['A0'] = X_train * drop_layer / drop_prob

	for i in range(1, num_layers):
		if batch_normalization:
			z = np.matmul(Weights['W' + str(i)], activations['A' + str(i - 1)])
		else:
			z = np.matmul(Weights['W' + str(i)], activations['A' + str(i - 1)]) + Weights['b' + str(i)]
		a = z
		z_norm = z
		if batch_normalization:
			mean = np.mean(z, axis=1, keepdims=True)
			variance = np.var(z, axis=1, keepdims=True)
			sqrtvar = np.sqrt(variance + 1e-5)
			ivariance = 1. / sqrtvar
			xmu = z - mean
			x_norm = xmu * ivariance
			z_norm = Weights["gamma" + str(i)] * x_norm + Weights["beta" + str(i)]
			activations['Xnorm' + str(i)] = x_norm
			activations['Ivar' + str(i)] = ivariance
			activations['Xmu' + str(i)] = xmu
		if (sigmoids[i] == 'relu'):
			a = relu(z_norm)
		elif (sigmoids[i] == 'softmax'):
			a = softmax(z_norm)
			np.clip(a, 1e-10, 0.9999999999, out=a)
		elif (sigmoids[i] == 'sigmoid'):
			a = sigmoid(z_norm)
		else:
			print (" Error Activation in input layer is not defined ")

		if testing == False:
			if i == num_layers - 1:
				drop_layer = np.random.binomial(1, 1, a.shape)
				activations['D' + str(i)] = drop_layer
				activations['drop_prob' + str(i)] = 1
			else:
				drop_layer = np.random.binomial(1, drop_prob, a.shape)
				a = a * drop_layer / drop_prob
				activations['drop_prob' + str(i)] = drop_prob
				activations['D' + str(i)] = drop_layer
		else:
			if i != num_layers - 1:
				a = a * drop_prob

		activations['A' + str(i)] = a
		activations['Z' + str(i)] = z_norm
		activations['D' + str(i)] = drop_layer

	return activations


def total_backpropagation(string, Weights, activations, Y_train, num_layers, node_list, sigmoids, batch_normalization):
	m = Y_train.shape[1]
	classes = Y_train.shape[0]
	dWeights = {}
	ti = Y_train
	yi = activations['A' + str(num_layers - 1)]

	if string == 'Cross':
		delC = - np.divide(ti, yi)
		deriv_sigma = np.zeros((classes, classes, m))

		for i in range(0, classes):
			for j in range(0, classes):
				if (i == j):
					deriv_sigma[i, j, :] = yi[i, :] * (1 - yi[i, :])
				else:
					deriv_sigma[i, j, :] = -yi[i, :] * yi[j, :]

		delC = delC.reshape([1, 9, m])
		delL = np.einsum('mnr,ndr->mdr', delC, deriv_sigma)
		delL = delL[0, :, :]
	elif string == 'MSE':
		delL = 2*(yi - ti) * yi * (1-yi)

	if batch_normalization:
		dWeights["dgamma" + str(num_layers - 1)] = np.sum(delL * activations['Xnorm' + str(num_layers - 1)], axis=1,
														  keepdims=True) / m
		dWeights["dbeta" + str(num_layers - 1)] = np.sum(delL, axis=1, keepdims=True) / m
		dxhat = delL * Weights["gamma" + str(num_layers - 1)]
		dvar = np.sum((dxhat * activations['Xmu' + str(num_layers - 1)] * (-0.5) * activations[
			"Ivar" + str(num_layers - 1)] ** 3), axis=1, keepdims=True)
		dmu = (np.sum((dxhat * -activations["Ivar" + str(num_layers - 1)]), axis=1, keepdims=True)) + (
		dvar * (-2.0 / m) * np.sum(activations['Xmu' + str(num_layers - 1)], axis=1, keepdims=True))
		dx1 = dxhat * activations["Ivar" + str(num_layers - 1)]
		dx2 = dvar * (2.0 / m) * activations['Xmu' + str(num_layers - 1)]
		dx3 = (1.0 / m) * dmu
		dx = dx1 + dx2 + dx3
		dWeights["dW" + str(num_layers - 1)] = np.matmul(dx, activations['A' + str(num_layers - 2)].T) / m
	else:
		dwL = np.matmul(delL, activations['A' + str(num_layers - 2)].T) / m
		dbL = np.sum(delL, axis=1, keepdims=True) / m
		dWeights['dW' + str(num_layers - 1)] = dwL
		dWeights['db' + str(num_layers - 1)] = dbL

	for l in range(num_layers - 2, 0, -1):
		delL = np.multiply(np.matmul(Weights['W' + str(l + 1)].T, delL), \
						   activation_derivative(activations['Z' + str(l)], sigmoids[l])) \
			   * activations['D' + str(l)]

		if batch_normalization:
			dWeights["dgamma" + str(l)] = np.sum(delL * activations['Xnorm' + str(l)], axis=1, keepdims=True) / m
			dWeights["dbeta" + str(l)] = np.sum(delL, axis=1, keepdims=True) / m
			dxhat = delL * Weights["gamma" + str(l)]
			dvar = np.sum((dxhat * activations['Xmu' + str(l)] * (-0.5) * activations[ \
				"Ivar" + str(l)] ** 3), axis=1, keepdims=True)
			dmu = (np.sum((dxhat * -activations["Ivar" + str(l)]), axis=1, keepdims=True)) + ( \
				dvar * (-2.0 / m) * np.sum(activations['Xmu' + str(l)], axis=1, keepdims=True))
			dx1 = dxhat * activations["Ivar" + str(l)]
			dx2 = dvar * (2.0 / m) * activations['Xmu' + str(l)]
			dx3 = (1.0 / m) * dmu
			dx = dx1 + dx2 + dx3
			dWeights["dW" + str(l)] = np.matmul(dx, activations['A' + str(l - 1)].T) / m

		else:
			dw = np.matmul(delL, activations['A' + str(l - 1)].T) / m
			db = np.sum(delL, axis=1, keepdims=True) / m
			dWeights['dW' + str(l)] = dw
			dWeights['db' + str(l)] = db

	return dWeights


## -----------------------------------------------------Model nn------------------------------------------------ ##
def model_nn(num_layers, regu, mini_batch_size, data, node_list, sigmoids, parameter_set = False, batch_normalization = False, drop_prob = 1):
	X_train,Y_train,X_test,Y_test = preprocess()
	num_batches = X_train.shape[1]/mini_batch_size
	X_batches = []
	Y_batches = []
	CVX_batches = []
	CVY_batches = []

	if parameter_set == False:
		batch_count = num_batches
	else:
		batch_count = num_batches - 40 * 128 / mini_batch_size

	if parameter_set == False:
		for i in range(0,num_batches):
			X_batches.append(X_train[:,i*mini_batch_size:(i+1)*mini_batch_size])
			Y_batches.append(Y_train[:,i*mini_batch_size:(i+1)*mini_batch_size])

	else:
		for i in range(0, batch_count):
			X_batches.append(X_train[:, i * mini_batch_size:(i + 1) * mini_batch_size])
			Y_batches.append(Y_train[:, i * mini_batch_size:(i + 1) * mini_batch_size])

		for i in range(batch_count, num_batches):
			CVX_batches.append(X_train[:, i * mini_batch_size:(i + 1) * mini_batch_size])
			CVY_batches.append(Y_train[:, i * mini_batch_size:(i + 1) * mini_batch_size])

		CVX_test = X_train[:, batch_count*mini_batch_size: num_batches*mini_batch_size]
		CVY_test = Y_train[:, batch_count*mini_batch_size: num_batches*mini_batch_size]

		X_train = X_train[:, 0:batch_count*mini_batch_size]
		Y_train = Y_train[:, 0:batch_count*mini_batch_size]

	epochs = 100
	Weights = initialize_weights(num_layers, node_list)
	learning_rate = 0.1
	regularizer = 'L1'
	regularization_factor = regu
	testaccuracy = 0
	training_cost = []
	test_cost = []
	test_accuracy = []
	training_cost_plot = []
	test_cost_plot = []
	test_accuracy_plot = []
	start_time = time.time()

	for i in range(0,epochs):
		for j in range(0, batch_count):
			X_mini_batch = X_batches[j]
			Y_mini_batch = Y_batches[j]
			activations = total_feed_forward(Weights, X_mini_batch, sigmoids, num_layers, batch_normalization, drop_prob, testing=False)
			dWeights = total_backpropagation('MSE', Weights, activations, Y_mini_batch, num_layers, node_list, sigmoids, batch_normalization)
			Weights = update_weights(Weights, dWeights, num_layers, learning_rate, regularizer, regularization_factor, batch_normalization)


		trainingactivations = total_feed_forward(Weights, X_train, sigmoids, num_layers, batch_normalization, 1, testing=True)
		trainingcost = crosscost('MSE', Y_train, trainingactivations['A' + str(num_layers - 1)], regularizer, regularization_factor, Weights, len(node_list)) / Y_train.shape[1]
		trainingaccuracy = accuracyC(Y_train, trainingactivations['A' + str(num_layers - 1)])

		if parameter_set == False:
			testactivations = total_feed_forward(Weights, X_test, sigmoids, num_layers, batch_normalization, 1, testing=True)
			testcost = crosscost('MSE', Y_test, testactivations['A'+str(num_layers-1)], regularizer, regularization_factor, Weights, len(node_list)) / Y_test.shape[1]
			testaccuracy = accuracyC(Y_test, testactivations['A'+str(num_layers-1)])
		else:
			testactivations = total_feed_forward(Weights, CVX_test, sigmoids, num_layers, batch_normalization, 1, testing=True)
			testcost = crosscost('MSE', CVY_test, testactivations['A' + str(num_layers - 1)], regularizer, regularization_factor, Weights, len(node_list)) / CVY_test.shape[1]
			testaccuracy = accuracyC(CVY_test, testactivations['A' + str(num_layers - 1)])

		training_cost_plot.append(trainingcost)
		test_cost_plot.append(testcost)
		test_accuracy_plot.append(testaccuracy * 100)
		data['Epoch'].append(i+1)
		data['Training Cost'].append(round(trainingcost, 3))
		data['Training Accuracy'].append(round(trainingaccuracy * 100, 3))
		data['Test Cost'].append(round(testcost, 3))
		data['Test Accuracy'].append(round(testaccuracy * 100, 3))

	elapsed = (time.time() - start_time) / epochs
	print("time taken per epoch ", round(elapsed, 4))
	plt.plot(range(0, len(training_cost_plot)), training_cost_plot, 'r', label='training error')
	plt.plot(range(0, len(test_cost_plot)), test_cost_plot, 'b', label='test error')
	plt.xlabel('Iterations')
	plt.ylabel('Average Cost')
	plt.legend()
	plt.show()


## ------------------------------------------------------------------------------------------------------------- ##
node_list = [784, 400, 80, 9]
num_layers = len(node_list)
sigmoids = ['NotApplied', 'sigmoid', 'sigmoid', 'softmax']
data = {'Epoch': [], 'Training Cost': [], 'Training Accuracy': [], 'Test Cost': [], 'Test Accuracy': []}

minibatch = [1, 32, 64, 128, 256, 512]
regularization = [0.0001, 0.0003, 0.001, 0.003, 0.01]
model_nn(num_layers, 0, 128, data, node_list, sigmoids, parameter_set=False, batch_normalization=False, drop_prob=1)
df = pd.DataFrame(data, columns=['Epoch', 'Training Cost', 'Training Accuracy', 'Test Cost', 'Test Accuracy'])
df.to_csv('MSE_sigmoid.csv', index=False, sep=',', encoding='utf-8')

