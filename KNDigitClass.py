from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
labels = pd.Series(digits['target'])
data = pd.DataFrame(digits['data'])

#values we re-use numerous times
k_values = np.arange(1,21,1) #used in knn models
n_splits = np.arange(4,13,2) #used in k-fold cross validation
neurons = [pow(2, x) for x in np.arange(2,9,1)] #used in nn models

#lists to hold resulting accuracy lists
model_results = []
model_names = []

#train a knn model
def train_knn(k, train_features, train_labels):
    knn = KNeighborsClassifier(k)
    knn.fit(train_features, train_labels)
    return knn

#test the accuracy of model
def test(model, test_features, test_labels):
    predictions = model.predict(test_features)
    return accuracy_score(test_labels, predictions)

#cross validates using KFold
def cross_validate(k, n_splits=4):
    train_acc = []
    test_acc = []
    kf = KFold(n_splits, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(data):
        #splitting data
        train_features, test_features = data.loc[train_index], data.loc[test_index]
        train_labels, test_labels = labels.loc[train_index], labels.loc[test_index]
        
        #train model
        model = train_knn(k, train_features, train_labels)
        
        #calculate accuracies
        train_accuracy = test(model, train_features, train_labels)
        test_accuracy = test(model, test_features, test_labels)
        
        #append accuracies to list
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        
    return np.mean(train_acc), np.mean(test_acc)
  
#function to help resuse code
def plot_knn_model(title, n_splits=4):
    #lists to contain accuracies of respective training/test sets
    fold_train_acc = {}
    fold_test_acc = {}
    train_acc = []
    test_acc = []
    
    #figure details
    fig, ax = plt.subplots(figsize=(12,8))
    fig.tight_layout(pad=3)
    fig.suptitle(y=0.9, t=f'Mean Accuracy vs. {title}', va='bottom', fontsize=22)
    
    if type(n_splits) != int:
        #loop to calculate accuracies with various k & n values
        for k in k_values:
            fold_train_acc[k] = []
            fold_test_acc[k] = []
            for n in n_splits:
                training, testing = cross_validate(k, n)        
                fold_train_acc[k].append(training)
                fold_test_acc[k].append(testing)        
            ax.plot(n_splits, fold_test_acc[k])
    else:
        #loop to calculate accuracies with various k values
        for k in k_values:
            training, testing = cross_validate(k)
            train_acc.append(training)
            test_acc.append(testing)

    #plot
    if type(n_splits) != int:
        ax.set_xticks(n_splits, fontsize=14)
        ax.set_xlabel('Number of Folds', fontsize=14)
        ax.legend(k_values, frameon=False, bbox_to_anchor=(1.02, 1), fontsize=14, title='k-neighbors')
    else:
        ax.plot(k_values, train_acc, linewidth=3, label='training')
        ax.plot(k_values, test_acc, linewidth=3, label='test')
        ax.set_xlabel('k-Neighbors', fontsize=14)
        ax.set_xticks(k_values)
        ax.tick_params(size=14)
        ax.legend(fontsize=14)        
    
    ax.set_ylabel('Accuracy', fontsize=14)    
    plt.show()
    
    if type(n_splits) != int:
        return fold_test_acc
    else:
        return test_acc
      
title = 'k-Neighbors'
KNN_results = plot_knn_model(title)
