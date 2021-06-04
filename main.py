import numpy as np 
import argparse
import sys
import pickle
import torch
import random
import itertools

import torch.utils.data as data
from torchvision.models import resnet

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# import data
import domain_net.wl as DWL
import domain_net.mini_domainnet as MDN
import aa2.aa2_data as AA2

# import algorithms
import algorithms.subgradient_method as SG
import algorithms.max_likelihood as ML

# baseline implementations
import labelmodels.labelmodels.naive_bayes as NB
import labelmodels.labelmodels.semi_supervised as SS
import all_algo as ALL
import heuristic_algo as PGMV

np.set_printoptions(threshold=sys.maxsize)

def str2bool(v):
	'''
	Used to help argparse library 
	'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def resnet_transform(unlabeled_data):
	'''
	Function to transform unlabeled data into features learned by 
	pre-trained resnet

	Args:
	unlabeled_data - raw pixel data
	'''

	ul1 = unlabeled_data[:300]
	ul2 = unlabeled_data[300:]
	res = resnet.resnet18(pretrained=True)
	td1 = res(torch.tensor(ul1))
	td1 = td1.detach().numpy()

	td2 = res(torch.tensor(ul2))
	td2 = td2.detach().numpy()

	transformed_data = np.concatenate([td1, td2])
	return transformed_data

def compute_avg_briar(y, y_pred, C):
	'''
	Function to compute the average briar loss over each example

	Args:
	y - true labels (one-hots)
	y_pred - prediction from the model (probability distribution)
	C - number of classes
	'''

	vals = []
	one_hots = np.eye(C)[y]

	# print(np.shape(y_pred), np.shape(one_hots))

	for i in range(len(y)):
		vals.append(SG.Brier_loss_linear(one_hots[i], y_pred[i]))

	return np.mean(vals)

def eval_comb(votes, labels, theta):
	'''
	Function to compute the accuracy of a weighted combination of labelers
	
	Args:
	votes - weak supervision source outputs
	labels - one hot labels
	theta - the weighting given to each weak supervision source
	'''

	N, M, C = np.shape(votes)
	totals = np.zeros((M, C))
	for i, val in enumerate(theta):
		for j, vote in enumerate(votes[i]):
			totals[j] += val * vote

	preds = np.argmax(totals, axis=1)
	briar_loss = compute_avg_briar(labels, totals, C)
	# print(np.mean(preds == true_labels))
	# print(confusion_matrix(true_labels, preds, labels=list(range(10))))
	return np.mean(preds == labels), briar_loss

def eval_lr(data, labels, theta, C):
	'''
	Function to evaluate a logistic regression model 

	Args:
	data - the data to evaluate the logreg model on
	labels - one hot labels
	theta - the weights for the logreg model
	C - the number of target classes
	'''

	probs = []
	preds = []
	for i, d in enumerate(data):
		p = SG.logistic_regression(theta, d)
		preds.append(np.argmax(p))
		probs.append(p)
	
	probs = np.array(probs)
	preds = np.array(preds)

	briar_loss = compute_avg_briar(labels, probs, C)
	return np.mean(preds == labels), briar_loss

def eval_sub(votes, sub):
	'''
	Function to evaluate the majority vote of a subset for binary tasks
	
	Args:
	votes - weak supervision source outputs
	sub - the subset of weak supervision sources to contribute to the majority vote
	'''

	N, M, C = np.shape(votes)
	probs = np.zeros((M, C))

	for i in sub:
		probs += votes[i]
	
	return probs / len(sub), np.argmax(probs, axis=1)


def compute_errors(votes, labels):
	'''
	Function to compute errors from votes of dimension (N, M, C)
	
	Args:
	votes - weak supervision source outputs
	labels - ground truth labels
	'''
	N, M, C = np.shape(votes)
	errors = np.zeros(N)
	for i in range(N):
		preds = np.argmax(votes[i], axis=1)
		errors[i] = np.mean(preds == labels)

	return errors

def correct_votes(errors, votes, vote_signals):
	'''
	Funciton to correct the votes for weak supervision sources with less than 50% accuracy
	(only in binary classificaiton tasks AwA2)

	Args:
	errors - error rates
	votes - weak supervision source outputs
	vote_signals - weak superivsion source outputs (soft classifications)
	'''
	for i, e in enumerate(errors):
		if e < 0.5:
			errors[i] = 1 - e
			votes[:, i] = np.where(votes[:, i] == 0, 1, 0)
			
			# swapping signal values
			M, C = np.shape(vote_signals[i])
			new_sigs = np.zeros((M, 2))
			new_sigs[:, 0] = vote_signals[i][:, 1]
			new_sigs[:, 1] = vote_signals[i][:, 0]
			vote_signals[i] = new_sigs
	return errors, votes, vote_signals

def convert_loaders(trainloader, testloader, unlab=500):
	'''
	Convert dataloaders into two new loaders, where the test loader has a specific
	number of examples

	Args:
	trainloader - training data loader
	testloader - testing data loader
	unlab - number of test data for new loader
	'''

	X = []
	Y = []
	
	for batch_x, batch_y in trainloader:
		for x in batch_x:
			X.append(x.numpy())
		for y in batch_y:
			Y.append(y.numpy())
	
	for batch_x, batch_y in testloader:
		for x in batch_x:
			X.append(x.numpy())
		for y in batch_y:
			Y.append(y.numpy())
	
	X = np.array(X)
	Y = np.array(Y)

	ratio = unlab / len(X)

	ss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=0)
	for lab_index, unlab_index in ss.split(X, Y):

		print(len(lab_index), len(unlab_index))

		unlab_X, unlab_Y = X[unlab_index], Y[unlab_index]
		lab_X, lab_Y = X[lab_index], Y[lab_index]

		lab_loader = data.DataLoader([(torch.tensor(lab_X[i]), torch.tensor(lab_Y[i])) for i in range(len(lab_X))], shuffle=False, batch_size=50)
		unlab_loader = data.DataLoader([(torch.tensor(unlab_X[i]), torch.tensor(unlab_Y[i])) for i in range(len(unlab_X))], shuffle=False, batch_size=50)
		return lab_loader, unlab_loader

def aa2_experiment(task_ind, lab=100, sg=False, baseline=False, ind=False, log_reg=False, pgmv=False, all_algo=False):
	'''
	Function to run experiments on the AwA2 dataset

	Args:
	task_ind - index of a binary AwA2 task 
	'''

	# hardest examples by avg majority vote accuracy less than 80%
	hard_experiments = [[5, 0.77], [6, 0.63], [11, 0.527], [12, 0.38] , [15, 0.725], [20, 0.508], [24, 0.68], [25, 0.516], [28, 0.321], [41, 0.366]] 
	sorted_experiments = sorted(hard_experiments, key=lambda x: x[1])
	print(sorted_experiments)

	comb_ind = sorted_experiments[task_ind - 1][0]

	# getting task information defined by start
	unseen_classes = AA2.get_test_classes()
	combs = list(itertools.combinations(range(10), 2))
	classes = combs[comb_ind - 1]
	unseen = [unseen_classes[classes[0]], unseen_classes[classes[1]]]
	task = str(classes[0]) + str(classes[1])
	print("Task: " + task)

	features = AA2.get_feature_diffs(unseen)
	(labeled_X, labeled_labels, train_names), (unlabeled_X, unlabeled_labels, test_names) = AA2.gen_unseen_data_split(classes, 0)

	# hard labelers
	# train_votes = AA2.get_votes(classes, features, train_names)
	# test_votes = AA2.get_votes(classes, features, test_names)

	# soft labelers
	labeled_votes = AA2.get_signals(classes, features, train_names)
	unlabeled_votes = AA2.get_signals(classes, features, test_names)

	N = len(features) # from five domains
	C = 2 # 5 classes in test sample

	# Unlabeled data and Labeled data
	num_lab = np.shape(labeled_votes)[1]
	num_unlab = np.shape(unlabeled_votes)[1]

	# use fraction of labeled data
	lab_indices = random.sample(list(range(num_lab)), lab)
	
	labeled_X = labeled_X[lab_indices]
	labeled_votes = labeled_votes[:, lab_indices]
	labeled_labels = labeled_labels[lab_indices]

	labeled_labels = np.eye(2)[labeled_labels - 1]
	unlabeled_labels = np.eye(2)[unlabeled_labels - 1]

	# print(np.shape(labeled_votes), np.shape(unlabeled_votes))

	num_lab = np.shape(labeled_votes)[1]
	print("Unlab: " + str(num_unlab) + " | Lab: " + str(num_lab))

	train_labels = np.argmax(labeled_labels, axis=1)
	tl = np.argmax(unlabeled_labels, axis=1)

	if log_reg:
		import warnings
		warnings.filterwarnings('ignore')
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(SG.cross_entropy_linear, 
																								unlabeled_votes, 
																								labeled_votes, 
																								labeled_labels)


		# SET EPS here
		eps = 0.3
		L = 2 * np.sqrt(N + 1)
		squared_diam = 2
		T = int(np.ceil(L*L*squared_diam/(eps*eps)))
		h = eps/(L*L)
		T = 2000

		# transforming data w/ Resnet
		transformed_data = resnet_transform(unlabeled_X)
		initial_theta = np.random.normal(0, 0.1, (len(transformed_data[0]), C))

		model_theta = SG.subGradientMethod(transformed_data, constraint_matrix, constraint_vector, 
										constraint_sign, SG.cross_entropy_linear, SG.logistic_regression, 
										SG.projectToBall,initial_theta, 
										T, h, N, num_unlab, C, lr=True)



		c = eval_lr(transformed_data, tl, model_theta, C)
		print("Subgradient LR: " + str(c)) # acc ,  briar loss


	if sg:
		import warnings
		warnings.filterwarnings('ignore')
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(SG.Brier_loss_linear, 
																								unlabeled_votes, 
																								labeled_votes, 
																								labeled_labels)

		# SET EPS here
		eps = 0.3
		L = 2 * np.sqrt(N + 1)
		squared_diam = 2
		T = int(np.ceil(L*L*squared_diam/(eps*eps)))
		h = eps/(L*L)
		T = 2000

		model_theta = SG.subGradientMethod(unlabeled_votes, constraint_matrix, constraint_vector, 
										constraint_sign, SG.Brier_loss_linear, SG.linear_combination_labeler, 
										SG.projectToSimplex, np.array([1 / N for i in range(N)]), 
										T, h, N, num_unlab, C)
		
		# evaluate learned model
		c = eval_comb(unlabeled_votes, tl, model_theta)
		print("Subgradient: " + str(c)) # acc ,  briar loss

	if baseline:

		# majority vote
		mv_preds = []
		mv_probs = []

		for i in range(num_unlab):
		
			vote = np.zeros(C)
			for j in range(N):
				# vote_val = np.argmax(unlabeled_votes[j][i])
				# vote[vote_val] += 1

				vote += unlabeled_votes[j][i]
			mv_preds.append(np.argmax(vote))
			mv_probs.append(vote / N)

		mv_probs = np.array(mv_probs)
		mv_acc = np.mean(mv_preds == tl)
		
		print("MV: " + str(mv_acc))
		print("MV Briar:" + str(compute_avg_briar(tl, mv_probs, C)))

		# semi-supervised ds
		wl_votes_test = np.zeros((num_unlab, N))
		for i in range(N):
			wl_votes_test[:, i] = np.argmax(unlabeled_votes[i], axis=1)

		wl_votes_train = np.zeros((num_lab, N))
		for i in range(N):
			wl_votes_train[:, i] = np.argmax(labeled_votes[i], axis=1)

		wl_votes_test += 1
		wl_votes_train += 1
		train_labels += 1

		ds_model = SS.SemiSupervisedNaiveBayes(C, N)

		# create SS dataset
		votes = np.concatenate((wl_votes_train, wl_votes_test))
		labels = np.concatenate((train_labels, np.zeros(num_unlab))).astype(int)
		
		ds_model.estimate_label_model(votes, labels)
		ds_preds = ds_model.get_most_probable_labels(wl_votes_test)
		ds_probs = ds_model.get_label_distribution(wl_votes_test)
		
		ds_acc = np.mean(ds_preds == tl + 1)
		print("DS: " + str(ds_acc))

		ds_briar = compute_avg_briar(tl, ds_probs, C)
		print("DS Briar: " + str(ds_briar))

	if ind:
		print("Individual Weak Labelers")
		
		results_dict = {}

		for i in range(N):
			wl_preds = np.argmax(unlabeled_votes[i], axis=1)
			results_dict[i] = (np.mean(wl_preds == tl), compute_avg_briar(tl, unlabeled_votes[i], C))

		# sort list by lowest 0-1 acc
		sor_res = sorted(results_dict.items(), key=lambda x: -x[1][0])
		for i in range(3):
			print(sor_res[i])
	

	if pgmv:
		error_estimates = compute_errors(labeled_votes, train_labels)

		# convert votes to single output value
		ul_votes = np.argmax(unlabeled_votes, axis=2).T

		error_estimates, ul_votes, unlabeled_votes = correct_votes(error_estimates, ul_votes, unlabeled_votes)
		print(np.shape(ul_votes))

		# run PGMV
		best_sub, best_ep = PGMV.heuristic_algo1(1, error_estimates, ul_votes, 5, 9)
		probs, preds = eval_sub(unlabeled_votes, best_sub)

		print("PGMV: " + str((np.mean(preds == tl), compute_avg_briar(tl, probs, C))))

	if all_algo:
		error_estimates = compute_errors(labeled_votes, train_labels)
		ul_votes = np.argmax(unlabeled_votes, axis=2).T

		# transforming data w/ Resnet
		transformed_data = resnet_transform(unlabeled_X)
		# initial_theta = np.random.normal(0, 0.1, (len(transformed_data[0]), C))

		# run ALL
		probs, preds, labels = ALL.eval_all_lr(transformed_data, ul_votes.T, tl, transformed_data, tl, error_estimates)
		print(preds, labels)
		print("ALL: " + str((np.mean(preds == labels), compute_avg_briar(labels, probs, C))))

def domain_net_experiment(test_domain, sample, lab=100, ml=False, sg=False, baseline=False, ind=False, sup=False):
	'''
	Function to run domain net experiments

	Args:
	test_domain - index of domain for target domain
	sample - number of sampled 5 classes to evaluate
	'''

	domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
	domains.remove(test_domain)

	# cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cuda0 = torch.device("cpu")

	# load domain net data
	trainloader, valloader, testloader = MDN.get_loaders(test_domain, sample)
	wls = DWL.get_weak_labelers(sample, domains)
	trained_wl = DWL.get_weak_labelers(sample, [test_domain])

	N = len(wls) # from five domains
	C = 5 # 5 classes in test sample

	# fixing data
	trainloader, testloader = convert_loaders(trainloader, testloader)

	# Load input and define important parameters 
	labeled_X, labeled_votes, labeled_labels = DWL.apply_wls(wls, trainloader, cuda0, soft=True)
	unlabeled_X, unlabeled_votes, unlabeled_labels = DWL.apply_wls(wls, testloader, cuda0, soft=True)  	

	_, tc_votes, tc_tl = DWL.apply_wls(trained_wl, testloader, cuda0, soft=True)


	# Unlabeled data and Labeled data
	num_lab = np.shape(labeled_votes)[1]
	num_unlab = np.shape(unlabeled_votes)[1]

	# use fraction of labeled data
	lab_indices = random.sample(list(range(num_lab)), lab)
	
	labeled_X = labeled_X[lab_indices]
	labeled_votes = labeled_votes[:, lab_indices]

	labeled_labels = labeled_labels[lab_indices]
	num_lab = np.shape(labeled_votes)[1]
	print("Unlab: " + str(num_unlab) + " | Lab: " + str(num_lab))

	train_labels = np.argmax(labeled_labels, axis=1)
	tl = np.argmax(unlabeled_labels, axis=1)
	tc_tl = np.argmax(tc_tl, axis=1)
	print(np.shape(labeled_labels))

	if sg:
		# import warnings
		# warnings.filterwarnings('ignore')
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(SG.Brier_loss_linear, 
																								unlabeled_votes, 
																								labeled_votes, 
																								labeled_labels)
		# SET EPS here
		eps = 0.3
		L = 2 * np.sqrt(N + 1)
		squared_diam = 2
		T = int(np.ceil(L*L*squared_diam/(eps*eps)))
		h = eps/(L*L)
		T = 2000

		model_theta = SG.subGradientMethod(unlabeled_votes, constraint_matrix, constraint_vector, 
										constraint_sign, SG.Brier_loss_linear, SG.linear_combination_labeler, 
										SG.projectToSimplex, np.array([1 / C for i in range(C)]), 
										T, h, N, num_unlab, C)
		
		# evaluate learned model
		c = eval_comb(unlabeled_votes, tl, model_theta)
		print("Subgradient: " + str(c)) # acc ,  briar loss

	if baseline:

		# majority vote
		mv_preds = []
		mv_probs = []

		for i in range(num_unlab):
		
			vote = np.zeros(C)
			for j in range(N):
				vote += unlabeled_votes[j][i]
			mv_preds.append(np.argmax(vote))
			mv_probs.append(vote / 5)

		mv_probs = np.array(mv_probs)
		# print(np.shape(mv_probs))

		mv_acc = np.mean(mv_preds == tl)
		print("MV: " + str(mv_acc))
		print("MV Briar:" + str(compute_avg_briar(tl, mv_probs, C)))

		# semi-supervised ds
		wl_votes_test = np.zeros((num_unlab, N))
		for i in range(N):
			wl_votes_test[:, i] = np.argmax(unlabeled_votes[i], axis=1)

		wl_votes_train = np.zeros((num_lab, N))
		for i in range(N):
			wl_votes_train[:, i] = np.argmax(labeled_votes[i], axis=1)

		wl_votes_test += 1
		wl_votes_train += 1
		train_labels += 1

		ds_model = SS.SemiSupervisedNaiveBayes(C, N)

		# create SS dataset
		votes = np.concatenate((wl_votes_train, wl_votes_test))
		labels = np.concatenate((train_labels, np.zeros(num_unlab))).astype(int)
		
		ds_model.estimate_label_model(votes, labels)
		ds_preds = ds_model.get_most_probable_labels(wl_votes_test)
		ds_probs = ds_model.get_label_distribution(wl_votes_test)
		
		ds_acc = np.mean(ds_preds == tl + 1)
		print("DS: " + str(ds_acc))

		ds_briar = compute_avg_briar(tl, ds_probs, C)
		print("DS Briar: " + str(ds_briar))

	if sup:
		sup_classifier = DWL.WeakLabeler(C, pretrained=True)

		sup_loader = data.DataLoader([(torch.tensor(labeled_X[i]), torch.tensor(np.argmax(labeled_labels[i]))) for i in range(num_lab)], shuffle=True, batch_size=100)
		DWL.train_weak_labeler(sup_classifier, sup_loader, valloader, cuda0)

		_, sup_votes, sup_tl = DWL.apply_wls([sup_classifier], testloader, cuda0)
		print("Supervised: " + str(np.mean(np.argmax(sup_votes[0], axis=1) == tc_tl)))

	if ind:
		print("Individual Weak Labelers")
		
		for i in range(N):
			wl_preds = np.argmax(unlabeled_votes[i], axis=1)
			print(np.mean(wl_preds == tl), compute_avg_briar(tl, unlabeled_votes[i], C))
	
	if ml:
		val, x  = ML.maximumLikelihood2(labeled_votes, labeled_labels, unlabeled_votes, ML.Brier_loss_linear)
				
		sol = np.zeros((num_unlab, C))
		for i in range(num_unlab):
			for j in range(C):
				sol[i, j] = x[i * C + j]
		
		ml_preds = np.argmax(sol, axis=1)
		ml_acc = np.mean(ml_preds == tl)
		print("Max Likelihood: " + str(ml_acc))


if __name__ == "__main__":
	# setting up argparsers
	parser = argparse.ArgumentParser()
	parser.add_argument('--domainnet', default=True, type=str2bool, help="if should run domainnet or AwA2 experiments")
	parser.add_argument('--domain', default=1, type=int, help="run script to compute wl stats")
	parser.add_argument('--sample', default=1, type=int, help="run script to compute wl stats")
	parser.add_argument('--baseline', default=False, type=str2bool, help="if should run with baselines")
	parser.add_argument('--ind', default=False, type=str2bool, help="if should run individual labelers")
	parser.add_argument('--ml', default=False, type=str2bool, help="if should run the maximum likelihood approach")
	parser.add_argument('--sg', default=False, type=str2bool, help="if should run the subgradient method")
	parser.add_argument('--sup', default=False, type=str2bool, help="if should run the supervised approach")
	parser.add_argument('--num_lab', default=100, type=int, help="what fraction of labeled data")
	parser.add_argument('--log', default=False, type=str2bool, help="logistic regression approach")

	# PGMV and ALL baselines for AwA2 binary case
	parser.add_argument('--pgmv', default=False, type=str2bool, help="PGMV binary approach")
	parser.add_argument('--all', default=False, type=str2bool, help="Adversarial Label Learning binary approach")

	args = parser.parse_args()

	# fixing seed
	seed = 0
	np.random.seed(seed)
	random.seed(0)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	if args.domainnet:
		domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
		test_domain = domains[args.domain - 1]
		print("Test Domain: " + test_domain)
		print("Sample: " + str(args.sample))
		domain_net_experiment(test_domain, args.sample, lab=args.num_lab, ml=args.ml, baseline=args.baseline, ind=args.ind, sup=args.sup, sg=args.sg)
	else:
		aa2_experiment(args.sample, lab=args.num_lab, baseline=args.baseline, ind=args.ind, sg=args.sg, log_reg=args.log, pgmv=args.pgmv, all_algo=args.all)