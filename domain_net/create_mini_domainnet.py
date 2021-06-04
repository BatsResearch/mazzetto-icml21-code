import random
import sys
import os
import pickle
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split

def get_classes(sample=1):
    '''
    Function to read the list of sampled classes

    Args:
    sample - which sample fo five classes to read
    '''

    with open("./domain_net/sample_classes_" + str(sample) + ".txt", "r") as f:
        l = f.readlines()
    
    return [x.strip() for x in l]

def sample_classes(num_classes, num_samples):
    '''
    Sample some classes to use in multi-class classification tasks

    Args:
    num_classes - the number of classes to sample
    the num_samples - the number of samples desired
    '''
 
    real_dir = "./domain_net/"
    class_names = os.listdir(os.path.join(real_dir, 'real'))
    random.seed(0)


    # count up min frequency of test class over all domains
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    min_freq = {}

    for c in class_names:
        min_len = float("inf")

        for d in domains:
            domain_len = len(os.listdir(os.path.join(real_dir, d, c)))
            if domain_len < min_len:
                min_len = domain_len

        min_freq[c] = min_len

    sorted_classes = sorted(min_freq.items(), key=lambda x:x[1])
    print(sorted_classes[-(num_classes * num_samples):])
    selected_classes = [x[0] for x in sorted_classes[-(num_classes * num_samples):]]
    random.shuffle(selected_classes)

    for i in range(num_samples):
        with open(os.path.join(real_dir, "sample_classes_" + str(i + 1) + ".txt"), 'w') as f:
            for c in selected_classes[num_classes * i : num_classes * (i + 1)]:
                f.write(c + "\n")

def create_pkl(domain, sample):
    '''
    Create pickle files for given domain and a sample of classes
    '''

    print(f'Creating pkl file for {domain}')
    data_dir = "./domain_net/"
    sample_dir = "./domain_net/sample_%d" % (sample)

    domain_dir = os.path.join(data_dir, domain)
    imgs = []
    labels = []
    
    classes = get_classes(sample)

    for class_name in classes:
        for img_path in os.listdir(os.path.join(domain_dir, class_name)):
            full_path = os.path.join(domain_dir, class_name, img_path)
            imgs.append(np.asarray(Image.open(full_path).convert('RGB').resize((84, 84)), dtype='uint8'))
            labels.append(class_name)

    imgs = np.asarray(imgs)
    
    # splitting into train, val, and test
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2
    
    train_dict = {'data': X_train, 'labels': y_train}
    val_dict = {'data': X_val, 'labels': y_val}
    test_dict = {'data': X_test, 'labels': y_test}

    with open(os.path.join(sample_dir, f'train_{domain}.pkl'), 'wb') as f:
        pickle.dump(train_dict, f)

    with open(os.path.join(sample_dir, f'val_{domain}.pkl'), 'wb') as f:
        pickle.dump(val_dict, f)

    with open(os.path.join(sample_dir, f'test_{domain}.pkl'), 'wb') as f:
        pickle.dump(test_dict, f)

if __name__ == '__main__':
    
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    for d in domains:
        create_pkl(d, 1)
        create_pkl(d, 2)