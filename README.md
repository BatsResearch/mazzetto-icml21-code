# Adversarial Multi Class Learning under Weak Supervision with Performance Guarantees - Code

## Getting Started

These instructions will setup our experiments to run on your local machine. 

### Installing 

In a virtual environment, please install the dependencies using 

```
pip install -r requirements.txt
```
Or alternatively
```
conda install --file requirements.txt
```

First, you will need to clone the BatsResearch/amcl Github repository. It serves as reference implementations of the algorithm proposed in our paper. This can be found at https://github.com/BatsResearch/amcl.  

Next, you will need the BatsResearch/labelmodels/ Github repository, which you can find here: https://github.com/BatsResearch/labelmodels. We use this repository for the Semi-Supervised Dawid Skene implementation as a baseline. You should clone the repository, making a "labelmodels/" directory. After installing the respository, you will need to switch to the semi branch of labelmodels by running 
```
git checkout semi
```

from the "labelmodels/" directory.


## Setup

Our experiments are performed on the DomainNet dataset and the Animals with Attributes 2 (AwA2) dataset. 

### DomainNet

To setup our experiments, you will first need to download the DomainNet images. You can find the links to download at http://ai.bu.edu/M3SDA/. You should place the unzipped directories for each domain (clipart, inforgraph, etc.) in the "domain_net" directory. Then, to process these images into more manageable pickle files, you can run
```
python create_mini_domainnet.py
```

from the "domain_net/" directory. This will create directories "domain_net/sample_1" and "domain_net/sample_2" with pickle files for train and test data for each of the 5 domains. After processing all of the data, we next can generate the weak supervision sources for each domain. You can do this by finetuning a ResNet on each of the domains, by running the commands
```
python wl.py --sample sample_index --ind domain_index 
```
and passing parameters sample_index to represent which sample of classes and domain_index to specify which particular domain (sorted alphabetically). We recommend to train these models in parallel for efficiency.

### Animals with Attributes 2

You can additionally setup the experiments on the AwA2 dataset by following the steps outlined in https://github.com/BatsResearch/mazzetto-aistats21-code. By following the steps in this repository, you can create the weak supervision sources and download the AwA2 dataset images.   

After creating the sources and their outputs, you should put the data in "aa2" to create the "aa2/Animals_with_Attributes2/" directory. You should put the votes and signals in "aa2/votes" and "aa2/signals" respectively. You should put the processed data files in a "aa2/data" directory.


## Running Experiments

After creating the weak supervision sources and storing their outputs, you can start running our experiments. You can run the experiments that make up the table in our paper by running 

```
python main.py 
```

Both of these scripts have various flags that you can pass in to run the variants of the datasets, particular samples or tasks, and the baselines.

## Citation

Please cite the following paper if you use our work. Thanks!

Alessio Mazzetto*, Cyrus Cousins*, Dylan Sam, Stephen H. Bach, and Eli Upfal. "Adversarial Multi Class Learning under Weak Supervision with Performance Guarantees". International Conference on Machine Learning (ICML), 2021.

```
@inproceedings{amcl,
  title = {Adversarial Multi Class Learning under Weak Supervision with Performance Guarantees}, 
  author = {Mazzetto, Alessio and Cyrus, Cousins, and Sam, Dylan and Bach, Stephen H., and Upfal, Eli}, 
  booktitle = {International Conference on Machine Learning (ICML)}, 
  year = 2021, 
}
```