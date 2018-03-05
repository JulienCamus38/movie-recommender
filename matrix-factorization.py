# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:54 2018

@author: camusj
"""

# Imports
try:
    import numpy as np
    np.random.seed(0)
except ImportError:
    raise ImportError('numpy module needs to be imported.')
    
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError:
    raise ImportError('sklearn.metrics module needs to be imported.')
                      
try:
    import parser
except ImportError:
    raise ImportError('parser module needs to be imported.')
                      
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('matplotlib.pyplot module needs to be imported.')
                      
try:
    import seaborn as sns
    sns.set()
except ImportError:
    raise ImportError('seaborn module needs to be imported.')
    
try:
    import pandas as pd
except ImportError:
    raise ImportError('pandas module needs to be imported.')


def get_mae(pred, actual):
    """
    Mean absolute error between predicted and actual arrays
    
    Arguments
    =========
    - pred : (ndarray)
        Array of predicted values
        
    - actual : (ndarray)
        Array of actual values
    """
    
    # Ignore nonzero terms
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_absolute_error(pred, actual)
    
    
def get_mse(pred, actual):
    """
    Mean squared error between predicted and actual arrays
    
    Arguments
    =========
    - pred : (ndarray)
        Array of predicted values
        
    - actual : (ndarray)
        Array of actual values    
    """
    
    # Ignore nonzero terms
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)
    

def plot_learning_curve(iter_array, model):
    """
    Learning curve of the model regarding the number of iterations
    
    Arguments
    =========
    - iter_array : (ndarray)
        Array of number of iterations
    
    - model : (dict)
        Dictionary with parameters representing the model
    """
    
    plt.plot(iter_array, model.train_mse, label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, label='Testing', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    
    plt.xlabel('Iterations', fontsize=25);
    plt.ylabel('MSE', fontsize=25);
    
    plt.legend(loc='best', fontsize=15);
    

class mf():
    def __init__(self, 
                 ratings,
                 K=5,
                 lmbdaV=0.0, 
                 lmbdaU=0.0,
                 lmbdaBV=0.0,
                 lmbdaBU=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Arguments
        =========
        - ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        - K : (int)
            Number of latent factors to use in matrix 
            factorization model
        
        - lmbdaV : (float)
            Regularization term for item latent factors
        
        - lmbdaU : (float)
            Regularization term for user latent factors
            
        - lmbdaBV : (float)
            Regularization term for item biases
        
        - lmbdaBU : (float)
            Regularization term for user biases
        
        - verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.ratings = ratings
        self.nb_users, self.nb_items = ratings.shape
        self.K = K
        self.lmbdaV = lmbdaV
        self.lmbdaU = lmbdaU
        self.lmbdaBV = lmbdaBV
        self.lmbdaBU = lmbdaBU
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.nb_samples = len(self.sample_row)
        self._v = verbose


    def train(self, nb_iter=1e2, eta=1e-3):
        """
        Train model for nb_iter iterations from scratch
        
        Arguments
        =========
        - nb_iter : (int)
            Number of iterations for training the model
            
        - eta : (float)
            Learning rate
        """
        
        # Initialize latent vectors        
        self.U = np.random.normal(scale=1./self.K, size=(self.nb_users, self.K))
        self.V = np.random.normal(scale=1./self.K, size=(self.nb_items, self.K))
        
        self.eta = eta
        self.user_bias = np.zeros(self.nb_users)
        self.item_bias = np.zeros(self.nb_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
        if self._v:  
            print('Global bias = {}'.format(self.global_bias))
            
        self.partial_train(nb_iter)
    
    
    def partial_train(self, nb_iter):
        """ 
        Train model for nb_iter iterations. Can be 
        called multiple times for further training
        
        Arguments
        =========
        - nb_iter : (int)
            Number of iterations for training the model
        """
        
        # Loop over the iterations
        for it in range(1, nb_iter+1):
            
            # Print some information about the current iteration
            if it % 10 == 0 and self._v:
                print('\tCurrent iteration: {}'.format(it))
              
            # Perform SGD algorithm
            self.training_indices = np.arange(self.nb_samples)
            np.random.shuffle(self.training_indices)
            self.sgd()
            

    def sgd(self):
        """ Stochastic gradient descent algorithm """
        
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            
            # Prediction
            prediction = self.predict(u, i)
            
            # Error
            e = (self.ratings[u, i] - prediction)
            
            # Update biases
            self.user_bias[u] += self.eta * (e - self.lmbdaBU * self.user_bias[u])
            if np.isnan(self.user_bias[u]):
                self.user_bias[u] = 0.0
                
            self.item_bias[i] += self.eta * (e - self.lmbdaBV * self.item_bias[i])
            if np.isnan(self.item_bias[i]):
                self.item_bias[i] = 0.0
            
            # Update latent factors
            self.U[u, :] += self.eta * (e * self.V[i, :] - self.lmbdaU * self.U[u,:])
            self.V[i, :] += self.eta * (e * self.U[u, :] - self.lmbdaV * self.V[i,:])
               
                      
    def predict(self, u, i):
        """
        Single user and item prediction

        Arguments
        =========
        - u : (int)
            User considered for the prediction
            
        - i : (int)
            Item considered for the prediction
        """
        
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.U[u, :].dot(self.V[i, :].T)
        return prediction
    
    
    def predict_all(self):
        """ Predict ratings for every user and item """
        
        # Initialization
        predictions = np.zeros((self.U.shape[0], self.V.shape[0]))
        
        # Loop over users
        for u in range(self.U.shape[0]):
            # Loop over items
            for i in range(self.V.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    
    
    def calculate_learning_curve(self, iter_array, test, eta=1e-3):
        """
        Keep track of mse as a function of training iterations
        
        Arguments
        =========
        - iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
            
        - test : (2D ndarray)
            Testing dataset (assumed to be user x item)
            
        - eta : (float)
            Learning rate
        
        Output
        ======
        The function creates two new class attributes:
        
        - train_mse : (list)
            Training data mse values for each value of iter_array
            
        - test_mse : (list)
            Test data mse values for each value of iter_array
        """
        
        # Initialization
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0
        
        # Loop over iterations
        for (i, nb_iter) in enumerate(iter_array):
            
            # Print some information about the iterations
            if self._v:
                print('Number of iterations: {}'.format(nb_iter))
                
            # Training process
            if i == 0:
                self.train(nb_iter - iter_diff, eta)
            else:
                self.partial_train(nb_iter - iter_diff)

            # Predictions
            predictions = self.predict_all()

            # Update train and test mse
            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            
            # Print some information about the mse
            if self._v:
                print('Train mse: ' + str(self.train_mse[-1]))
                print('Test mse: ' + str(self.test_mse[-1]))
                
            # Update iter_diff
            iter_diff = nb_iter
           
           
# Main function
if __name__ == '__main__':
    
    # Parse the CSV file
    p = parser.parser()
    p.parse_csv()
    
    # Split into a train and a test arrays
    train, test = p.train_test_split()
    
    # Find the optimal hyperparameters
    iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
    '''
    etas = [1e-5, 1e-4, 1e-3, 1e-2]
    
    best_params = {}
    best_params['eta'] = None
    best_params['nb_iter'] = 0
    best_params['train_mse'] = np.inf
    best_params['test_mse'] = np.inf
    best_params['model'] = None
    
    for eta in etas:
        print('Learning rate (eta): {}'.format(eta))
        res = mf(train, K=5)
        res.calculate_learning_curve(iter_array, test, eta=eta)
        min_idx = np.argmin(res.test_mse)
        if res.test_mse[min_idx] < best_params['test_mse']:
            best_params['nb_iter'] = iter_array[min_idx]
            best_params['eta'] = eta
            best_params['train_mse'] = res.train_mse[min_idx]
            best_params['test_mse'] = res.test_mse[min_idx]
            best_params['model'] = res
            print('New optimal hyperparameters')
            print(pd.Series(best_params))
    '''
    eta = 1e-3        
    K_array = [5, 10, 20, 40, 80]
    lmbdas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    lmbdas.sort()
    
    best_params = {}
    best_params['K'] = K_array[0]
    best_params['lambda'] = lmbdas[0]
    best_params['nb_iter'] = 0
    best_params['train_mse'] = np.inf
    best_params['test_mse'] = np.inf
    best_params['model'] = None
    
    for K in K_array:
        print('Number of latent factors (K): {}'.format(K))
        for lmbda in lmbdas:
            print('Regularization parameter (lambda): {}'.format(lmbda))
            res = mf(train, K=K, lmbdaU=lmbda, lmbdaV=lmbda, lmbdaBU=lmbda, 
                     lmbdaBV=lmbda)
            res.calculate_learning_curve(iter_array, test, eta=eta)
            min_idx = np.argmin(res.test_mse)
            if res.test_mse[min_idx] < best_params['test_mse']:
                best_params['K'] = K
                best_params['lambda'] = lmbda
                best_params['nb_iter'] = iter_array[min_idx]
                best_params['train_mse'] = res.train_mse[min_idx]
                best_params['test_mse'] = res.test_mse[min_idx]
                best_params['model'] = res
                print('New optimal hyperparameters')
                print(pd.Series(best_params))
    
    # Plot
    plot_learning_curve(iter_array, best_params['model'])
    
    # Print information
    print('Best regularization (lambda): {}'.format(best_params['lambda']))
    print('Best latent factors (K): {}'.format(best_params['K']))
    print('Best iterations: {}'.format(best_params['nb_iter']))