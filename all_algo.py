import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import resnet
import torchvision.transforms as transforms

import os
import sys
import argparse
import distutils
import pickle
import itertools

# setting random seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OneLayer(nn.Module):
	'''
	Class representing a one layer neural network to be used as LR with soft labels
	on the votes matrix
	'''

	def __init__(self, num_inputs):
		super(OneLayer, self).__init__()
		self.out_layer = nn.Linear(num_inputs, 2)
		self.sigmoid = nn.Sigmoid()

	def forward(self, votes_matrix):
		return self.out_layer(votes_matrix)

def compute_probs(model, data, batch_size=50):
	'''
	Function to compute the probabilities of each class given the current parameters of the model
	
	Args:
	model - model to evaluate
	data - torch Tensor containing data
	'''

	probs = []
	n = len(data)

	for batch_ind in range(0, n, batch_size):
		batch_data = data[batch_ind : min(batch_ind + 50, n)]
		
		probs.append(model(batch_data))

	probs = torch.cat(probs)		
	probs = F.softmax(probs, dim=1)
	return probs

def y_gradient(y, p, q, b, rho, gamma):
	"""
	Computes the udpate to the adversarial labels
	"""
	
	n = p.size

	learnable_term = 1 - (2 * p)
	learnable_term = np.sum(learnable_term, axis=0) / n

	ls_term = 1 - (2 * q)
	gamma_term = np.dot(gamma.T, ls_term) / n

	weak_term = np.dot(1 - q, y) + np.dot(q, 1 - y)
	ineq_constraint = (weak_term / n) - b
	ineq_constraint = ineq_constraint.clip(min=0)
	ineq_augmented_term = rho * np.dot(ineq_constraint.T, ls_term)

	return learnable_term + gamma_term - ineq_augmented_term

def gamma_gradient(y, q, b):
	"""
	Computes the gradient of lagrangian inequality penalty parameters
	:param y: vector of estimated labels for the data
	:type y: array
	:param weak_signal_probabilities: soft or hard weak estimates for the data labels
	:type weak_signal_probabilities: ndarray, size no of weak signals x no of datapoints
	:param weak_signal_ub: upper bounds error rates for the weak signals
	:type weak_signal_ub: array
	:return: vector of length gamma containing the gradient of gamma
	:rtype: array
	"""
	_, n = q.shape

	weak_term = np.dot(1 - q, y) + np.dot(q, 1 - y)
	ineq_constraint = (weak_term / n) - b

	return ineq_constraint

def eval_all(train_data, weak_signals, train_labels, test_data, test_labels, bounds, max_iter=300, device=cuda0):
	"""
	Function to train an end model using Adversarial Label Learning
	
	Note:
	labels are only used in evaluating progress of training, not in training

	"""

	print("Training on device: " + str(device))
	labels = train_labels - 1
	m,n = np.shape(weak_signals)

	print("Num of datapoints: " + str(n))

	# initialize algorithm variables (as done in original ALL paper)
	y = 0.5 * np.ones(n)
	gamma = np.zeros(weak_signals.shape[0])
	one_vec = np.ones(n)

	rho = 2.5
	lr = 0.0001

	t = 0
	converged = False

	model = OneLayer(m)
	learn_params = []
	
	for name, param in model.named_parameters():
		if param.requires_grad:
			learn_params.append(param)

	optimizer = torch.optim.Adam(learn_params, lr=0.0001)
	loss_function = nn.BCELoss()

	np_data = test_data
	np_labels = test_labels - 1

	# moving to GPU/CPU
	train_data = torch.tensor(train_data)
	train_data = train_data.to(device)
	model = model.to(device)

	probs = compute_probs(model, train_data, batch_size=50)
	probs = probs.cpu().detach().numpy()
	probs = probs[:,1]

	while not converged and t < max_iter:
		
		rate = 1 / (1 + t)

		# update y
		old_y = y

		y_grad = y_gradient(y, probs, weak_signals, bounds, rho, gamma)
		y = y + rate * y_grad

		# projection step: clip y to [0, 1]
		y = y.clip(min=0, max=1)

		# compute gradient of probabilities
		dl_dp = (1 / n) * (1 - 2 * old_y)

		# update gamma
		old_gamma = gamma
		gamma_grad = gamma_gradient(old_y, weak_signals, bounds)
		gamma = gamma - rho * gamma_grad
		gamma = gamma.clip(max=0)
		
		y_lab = torch.tensor(y)
		y_lab = y_lab.float()

		# update model
		optimizer.zero_grad()

		y_lab = y_lab.to(device)

		output = compute_probs(model, train_data)[:,1]

		loss = loss_function(output, y_lab)

		loss.backward()
		optimizer.step()

		output = output.to("cpu")
		probs = output.detach().numpy()
		conv_y = np.linalg.norm(y - old_y)

		# check that inequality constraints are satisfied
		ineq_constraint = gamma_gradient(y, weak_signals, bounds)
		ineq_infeas = np.linalg.norm(ineq_constraint.clip(min=0))

		converged = np.isclose(0, conv_y, atol=1e-6) and np.isclose(0, ineq_infeas, atol=1e-6)

		if t % 30 == 0:
			print("Update %d" % (t))
			eval_model(np_data, np_labels + 1, model.to("cpu"), cuda0)
			model.to(device)
		t += 1

	print("Final Accuracy:")
	return eval_model(np_data, np_labels + 1, model.to("cpu"), cuda0)

def eval_all_lr(train_data, weak_signals, train_labels, test_data, test_labels, bounds, max_iter=3000, device=cuda0):
	"""
	Function to train an end model using Adversarial Label Learning
	
	Note:
	labels are only used in evaluating progress of training, not in training
	"""

	print("Training on device: " + str(device))
	labels = train_labels - 1
	m,n = np.shape(weak_signals)



	print("Num of datapoints: " + str(n))

	# initialize algorithm variables (as done in original ALL paper)
	y = 0.5 * np.ones(n)
	gamma = np.zeros(weak_signals.shape[0])
	one_vec = np.ones(n)

	rho = 2.5
	lr = 0.0001

	t = 0
	converged = False

	model = OneLayer(np.shape(train_data)[1])
	learn_params = []
	
	for name, param in model.named_parameters():
		if param.requires_grad:
			learn_params.append(param)

	optimizer = torch.optim.Adam(learn_params, lr=0.0001)
	loss_function = nn.BCELoss()

	np_data = test_data
	np_labels = test_labels - 1

	# moving to GPU/CPU
	train_data = torch.tensor(train_data).float()
	train_data = train_data.to(device)
	model = model.to(device)

	probs = compute_probs(model, train_data, batch_size=50)
	probs = probs.cpu().detach().numpy()
	probs = probs[:,1]

	while not converged and t < max_iter:
		
		rate = 1 / (1 + t)

		# update y
		old_y = y

		y_grad = y_gradient(y, probs, weak_signals, bounds, rho, gamma)
		y = y + rate * y_grad

		# projection step: clip y to [0, 1]
		y = y.clip(min=0, max=1)

		# compute gradient of probabilities
		dl_dp = (1 / n) * (1 - 2 * old_y)

		# update gamma
		old_gamma = gamma
		gamma_grad = gamma_gradient(old_y, weak_signals, bounds)
		gamma = gamma - rho * gamma_grad
		gamma = gamma.clip(max=0)
		
		y_lab = torch.tensor(y)
		y_lab = y_lab.float()

		# update model
		optimizer.zero_grad()

		y_lab = y_lab.to(device)

		output = compute_probs(model, train_data)[:,1]
		loss = loss_function(output, y_lab)

		loss.backward()
		optimizer.step()

		output = output.to("cpu")
		probs = output.detach().numpy()
		conv_y = np.linalg.norm(y - old_y)

		# check that inequality constraints are satisfied
		ineq_constraint = gamma_gradient(y, weak_signals, bounds)
		ineq_infeas = np.linalg.norm(ineq_constraint.clip(min=0))

		converged = np.isclose(0, conv_y, atol=1e-6) and np.isclose(0, ineq_infeas, atol=1e-6)
		if t % 1000 == 0:
			print("Update %d" % (t))
			eval_model(train_data, np_labels + 1, model.to("cpu"), cuda0)
			model.to(device)
		t += 1

	return eval_model(train_data, np_labels + 1, model.to("cpu"), cuda0)

def eval_model(test_data, test_labels, end_model, device): 
	'''
	Function to evaluate the end model 
	'''

	dl = data.DataLoader(list(zip(test_data, test_labels)), shuffle=True, batch_size=100, num_workers=0)
	probs = []
	learned_preds = []
	ls = []
	
	end_model = end_model.to(device)

	for d in dl:

		x, y = d

		# moving data to GPU/CPU
		inputs_batch = x.to(device)
		outputs = end_model(inputs_batch)
		_, end_preds = torch.max(outputs, 1)
		p = F.softmax(outputs, dim=1).cpu().detach().numpy()
		end_preds = end_preds.cpu().detach().numpy()
		y = y.detach().numpy()
		
		probs.append(p)
		learned_preds.append(end_preds)
		ls.append(y)

	probs = np.concatenate(probs)
	learned_preds = np.concatenate(learned_preds)
	learned_preds = np.where(learned_preds == 1, 0, 1)
	ls = np.concatenate(ls)

	# flipping probs
	new_probs = np.zeros_like(probs)
	new_probs[:, 0] = probs[:, 1]
	new_probs[:, 1] = probs[:, 0]
		
	return probs, learned_preds, ls