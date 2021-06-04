import os
from glob import glob
from PIL import Image
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# filepaths for data locations
data_path = "./aa2/Animals_with_Attributes2/"
classes_path = data_path + "classes.txt"
test_path 	= data_path + "testclasses.txt"
attributes_path = data_path + "predicates.txt"
images_path = data_path + "JPEGImages"

def get_classes():
	'''
	Function to create a mapping from class name to index
	'''
	
	class_to_index = {}

	with open(classes_path) as f:
		index = 0
		for line in f:
			class_name = line.split('\t')[1].strip()
			class_to_index[class_name] = index
			index += 1
	
	return class_to_index

def get_attributes():
	'''
	Function to get the list of attributes of the AA2 dataset
	'''

	attributes = []
	with open(attributes_path) as a_f:
		for line in a_f.readlines():
			attribute = line.strip().split()[1]
			attributes.append(attribute)
	return attributes

def create_attribute_matrix():
	'''
	Function to create a matrix of classes and attributes
	'''

	predicate_binary_mat = np.array(np.genfromtxt(data_path + "predicate-matrix-binary.txt", dtype='int'))
	return predicate_binary_mat

attributes = get_attributes()
attributes_matrix = create_attribute_matrix()
classes_map = get_classes()

def get_test_classes():
	'''
	Function to get the test classes
	'''
	test_classes = []
	with open(test_path) as test_f:
		test_classes = [x.strip() for x in test_f.readlines()]
	return test_classes

def get_feature_diffs(unseen_classes):
	'''
	Function to generate the feature differences between two unseen classes

	Args:
	unseen_classes - a list of two strings that represent the unseen class name
	'''

	binary_class_indices = [classes_map[unseen_classes[0]], classes_map[unseen_classes[1]]]
	binary_class_features = []

	atts_1 = attributes_matrix[binary_class_indices[0]]
	atts_2 = attributes_matrix[binary_class_indices[1]]
	for i in range(85):
		if atts_1[i] != atts_2[i]:
			if atts_1[i] == 1:
				binary_class_features.append(attributes[i])
			else:
				binary_class_features.append("!" + attributes[i])
	return binary_class_features

class AA2_Dataset(data.dataset.Dataset):
	'''
	Class for the Animals with Attributes 2 Dataset
	'''

	def __init__(self, classes, transform=None):
		self.class_map = get_classes()
		self.predicate_binary_mat = create_attribute_matrix()

		# looping through directory to get 
		self.image_names = []
		self.image_indices = []
		self.transform = transform
		self.image_labels = []

		for c in classes:
			FOLDER_DIR = os.path.join(images_path, c)
			file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
			files = glob(file_descriptor)

			for file_name in files:
				self.image_names.append(file_name)
				self.image_indices.append(self.class_map[c])
				self.image_labels.append(classes.index(c))
	
	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, index):
		'''
		Function to get an image
		'''

		im = Image.open(self.image_names[index])
		im_copy = im
		
		if im.getbands()[0] == 'L':
			im_copy = im_copy.convert('RGB')
		
		if self.transform:
			im_copy = self.transform(im_copy)
		im_array = np.array(im_copy)
		im_index = self.image_indices[index]
		im_predicate = self.predicate_binary_mat[im_index,:]
		im.close()

		return im_array, im_predicate, self.image_labels[index], self.image_names[index]


def get_trainloader(batch_size, shuffle=True):
	'''
	Function to get a trainloader for the AA2 dataset
	'''

	classes = get_train_classes()

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	])

	train_dataset = AA2_Dataset(classes, transform=transform)
	trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
	print("train length: " + str(len(trainloader)))
	return trainloader


def get_valloader(batch_size, shuffle=True):
	'''
	Function to get a valloader for the AA2 dataset
	'''

	classes = get_val_classes()

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	])

	val_dataset = AA2_Dataset(classes, transform=transform)
	valloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
	print("val length: " + str(len(valloader)))
	return valloader

def get_testloader(batch_size, shuffle=False):
	'''
	Function to get a testloader for the AA2 dataset
	'''

	classes = get_test_classes()

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	])

	test_dataset = AA2_Dataset(classes, transform=transform)
	testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
	return testloader

def load_votes_data():

	# loading unseen class data
	unseen_data = np.load("./aa2/data/unseen_data.npy")
	unseen_lables = np.load("./aa2/data/unseen_labels.npy")	
	unseen_names = pickle.load(open("./aa2/data/unseen_names.p", "rb"))

	# loading all attribute detectors
	wls = list(range(85))
	votes_matrix = np.zeros((len(unseen_names), 85))

	for i in wls:
		votes_path = "./aa2/votes/wl_votes_%d.p" % (i)
		votes_dict = pickle.load(open(votes_path, "rb"))

		for j, n in enumerate(unseen_names):
			votes_matrix[j, i] = votes_dict[n]

	return votes_matrix

def gen_unseen_data_split(classes, seed):
	'''
	Creating unseen dataset with a train and test split for an inductive approach
	'''

	base_path = "./aa2/data/"

	unseen_data = np.load(base_path + "unseen_data.npy")
	unseen_attributes = np.load(base_path + "unseen_attributes.npy")
	unseen_labels = np.load(base_path + "unseen_labels.npy")
	unseen_names = np.array(pickle.load(open(base_path + "unseen_names.p", "rb")))

	valid_indices = np.concatenate([np.nonzero(unseen_labels == classes[0]), 
									np.nonzero(unseen_labels == classes[1])], axis=None) 

	unseen_data = unseen_data[valid_indices]
	unseen_labels = unseen_labels[valid_indices]
	unseen_attributes = unseen_attributes[valid_indices]
	unseen_names = unseen_names[valid_indices]

	# converting labels
	unseen_labels = np.where(unseen_labels == classes[0], 1, 2)

	# splitting to test and train data (w/ stratified)
	train_indices, test_indices = train_test_split(range(len(unseen_labels)), test_size=0.5, random_state=seed, stratify=unseen_labels)

	train_data, train_labels = unseen_data[train_indices], unseen_labels[train_indices]
	train_atts, train_names = unseen_attributes[train_indices], unseen_names[train_indices]

	test_data, test_labels = unseen_data[test_indices], unseen_labels[test_indices]
	test_atts, test_names = unseen_attributes[test_indices], unseen_names[test_indices]

	# splitting train data into labeled and unlabeled
	return (train_data, train_labels, train_names), (test_data, test_labels, test_names)

def get_votes(classes, features, names):
	'''
	Function to get a set of votes
	'''

	num_examples = len(names)
	num_wls = len(features)

	# constructing list of weak labelers to load/invert
	indices = []
	for feature in features:
		# checking for negation
		to_add = [True, 0]
		if feature[0] == "!":
			to_add[0] = False
		to_add[1] = attributes.index(feature.replace("!", ""))
		indices.append(to_add)

	votes_matrix = np.zeros((num_wls, num_examples, 2))

	for i, tup in enumerate(indices):

		vote_correct, index = tup
		vote_dict = pickle.load(open("./aa2/votes/wl_votes_%d.p" % (index), "rb"))
		
		for j, name in enumerate(names):
			to_assign = vote_dict[name]
			
			if to_assign == 1 and vote_correct:
				votes_matrix[i][j][1] = 1 
			elif to_assign == 1 and not vote_correct:
				votes_matrix[i][j][0] = 1 
			elif to_assign == 0 and vote_correct:
				votes_matrix[i][j][0] = 1 
			else:
				votes_matrix[i][j][1] = 1 
			
	return votes_matrix


def get_signals(classes, features, names):
	'''
	Function to get a set of signals
	'''

	num_examples = len(names)
	num_wls = len(features)

	# constructing list of weak labelers to load/invert
	indices = []
	for feature in features:
		# checking for negation
		to_add = [True, 0]
		if feature[0] == "!":
			to_add[0] = False
		to_add[1] = attributes.index(feature.replace("!", ""))
		indices.append(to_add)

	sig_matrix = np.zeros((num_wls, num_examples, 2))

	for i, tup in enumerate(indices):

		sig_correct, index = tup

		sig_dict = pickle.load(open("./aa2/signals/signals_%d.p" % (index), "rb"))
		
		for j, name in enumerate(names):
			sig = sig_dict[name]
			
			if sig_correct:
				sig = 1 - sig
			
			sig_matrix[i][j][1] = sig
			sig_matrix[i][j][0] = 1 - sig
				
	return sig_matrix


if __name__ == "__main__":
	print(get_test_classes())