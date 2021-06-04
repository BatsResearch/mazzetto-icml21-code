import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import resnet
import torchvision.transforms as transforms
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class WeakLabeler(nn.Module):
	'''
	Class for a weak labeler on the Domain Net dataset
	'''

	def __init__(self, num_outputs, pretrained=True):
		'''
		Constructor for an endmodel
		'''

		super(WeakLabeler, self).__init__()

		# use un-initialized resnet so that more distinct representations are learned
		self.model = resnet.resnet18(pretrained=pretrained)
		for param in self.model.parameters():
			param.requires_grad = not pretrained
		
		num_features = 1000 # number of output features for resnet	
		self.hidden_layer = nn.Linear(num_features, 48)
		self.out_layer = nn.Linear(48, num_outputs)	
		self.dropout_layer = nn.Dropout(p=0.5)

	def forward(self, images):
		'''
		Method for a foward pass of the endmodel
		'''

		resnet_out = self.model(images)
		return self.out_layer(self.dropout_layer(self.hidden_layer(resnet_out)))

def get_class_weighting(trainloader):
	'''
	Function to get a weighting for the loss function class frequency
	'''

	class_count = {}
	total = 0
	for _, y in trainloader:
		for l in y:
			l = l.item()
			if l in class_count:
				class_count[l] += 1
			else:
				class_count[l] = 1
			total += 1

	weights = np.ones(5)
	for c in class_count:
		weights[c] =  max(np.log(total / class_count[c]), 1.0)

	return torch.Tensor(weights)

def train_weak_labeler(model, trainloader, valloader, device, num_epochs=20, lr=0.001):
	'''
	Function to train weak labelers to detect a given attribute

	Args:
	model - the weak labeler to train
	trainloader - the train dataset for the weak labeler
	valloader - the validation dataset for the weak labeler
	device - the device to train the weak labeler on
	'''

	# print("Train Datapoints: " + str(len(trainloader) * 50))

	learn_params = []
	for name, param in model.named_parameters():
		if param.requires_grad:
			learn_params.append(param)
	
	weights = get_class_weighting(trainloader)
	optimizer = torch.optim.Adam(learn_params, lr=lr)
	objective_function = torch.nn.CrossEntropyLoss()	 	
	model.to(device)

	# looping for epochs
	for ep in range(num_epochs):

		total_ex = 0
		pos_ex = 0

		for x, y in trainloader:

			# zeroing gradient
			optimizer.zero_grad()
			# moving data to GPU/CPU
			inputs = x.to(device)
			labels = y.to(device)

			pos_ex += torch.sum(labels).item()
			total_ex += len(labels)

			outputs = model(inputs)
			loss = objective_function(outputs, labels)
			loss.backward()
			optimizer.step()

		
		if ep % 20 == 0:
			print("Epoch: %d" % (ep))
			print("Train Data")
			eval_weak_labeler(model, trainloader, device)
			print("Val Data")
			eval_weak_labeler(model, valloader, device)

def eval_weak_labeler(model, dataloader, device): 
	'''
	Function to evaluate a weak labeler on the test data

	Args:
	model - the weak labeler
	testloader - the test data to evaluate on
	device - the device to run the model on 
	'''
	
	model.eval()

	predictions = []
	labs = []
	cm = 0

	for x, y in dataloader:
		np_labels = y.numpy()
		
		# moving data to GPU/CPU
		inputs = x.to(device)
		labels = y.to(device)

		_, preds = torch.max(model(inputs), 1)
		acc = torch.sum(preds == labels).item() / x.size()[0]
		np_preds = torch.Tensor.cpu(preds).numpy()

		predictions.append(np_preds)
		labs.append(np_labels)

	predictions = np.concatenate(predictions)
	labs = np.concatenate(labs)
	
	print("Accuracy: %f" % (np.mean(predictions == labs)))

	model.train()

def get_weak_labelers(sample, domains):
	'''
	Functions to load all weak labelers from a given sample of test classes
	'''

	wls = []

	for i in domains:
		wl = WeakLabeler(5, pretrained=True)
		wl.load_state_dict(torch.load("./domain_net/weak_labelers/sample_%d/" % (sample) + i))
		wls.append(wl)
	
	return wls

def apply_wls(wls, dataloader, device, soft=True):
	'''
	Function to apply weak labelers
	'''

	X = []
	wl_predictions = []
	Y = []
	
	for i, model in enumerate(wls):
		predictions = []

		for x, y in dataloader:
			
			if i == 0:
				np_data = x.numpy()
				np_labels = y.numpy()

			# moving data to GPU/CPU
			model = model.to(device)
			inputs = x.to(device)
			labels = y.to(device)

			if soft:
				preds = F.softmax(model(inputs), dim=1)
				np_preds = preds.detach().numpy()
			
			else:
				_, preds = torch.max(model(inputs), 1)
				np_preds = torch.Tensor.cpu(preds).numpy()

			if i == 0:
				X.append(np_data)
				Y.append(np_labels)
			
			predictions.append(np_preds)

		if i == 0:
			X = np.concatenate(X)
			Y = np.eye(5)[np.concatenate(Y)]
		
		predictions = np.concatenate(predictions)

		if soft:
			wl_predictions.append(predictions)
		else:
			wl_predictions.append(np.eye(5)[predictions])

	return X, np.array(wl_predictions), Y 

if __name__ == '__main__':

	import mini_domainnet as MD
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', default=0.005, type=float, help="learning rate")
	parser.add_argument('--epochs', default=100, type=int, help="number of training epochs")
	parser.add_argument('--ind', default=0, type=int, help="index of domain to train")
	parser.add_argument('--pretrained', default=True, type=bool, help="index of domain to train")
	parser.add_argument('--sample', default=1, type=int, help="which class sample to train over")
	args = parser.parse_args()

	seed = 0
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
	domain = domains[args.ind - 1]
	print("Domain: %s" % (domain))

	num_classes = 5

	trainloader, valloader, testloader = MD.get_loaders(domain, args.sample)
	wl = WeakLabeler(num_classes, pretrained=args.pretrained)

	print(len(trainloader))
	sys.exit()

	# train weak labeler and check val results
	train_weak_labeler(wl, trainloader, valloader, cuda0, num_epochs=args.epochs, lr=args.lr)

	# eval on test data
	eval_weak_labeler(wl, testloader, cuda0)
	torch.save(wl.state_dict(), "./domain_net/weak_labelers/sample_%d/" % (args.sample) + domain) 

