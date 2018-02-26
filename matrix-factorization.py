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
    from sklearn.metrics import mean_absolute_error
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

def get_mae(pred, actual):
    """ Mean squared error between predicted and actual arrays."""
    
    # Ignore nonzero terms
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_absolute_error(pred, actual)

def plot_learning_curve(iter_array, model):
    """ Learning curve of the model regarding the number of iterations. """
    
    plt.plot(iter_array, model.train_mae, label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mae, label='Testing', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    
    plt.xlabel('Iterations', fontsize=25);
    plt.ylabel('MAE', fontsize=25);
    
    plt.legend(loc='best', fontsize=15);

class mf():
    def __init__(self, 
                 ratings,
                 K=5,
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        K : (int)
            Number of latent factors to use in matrix 
            factorization model
        
        item_fact_reg : (float)
            Regularization term for item latent factors
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            
        item_bias_reg : (float)
            Regularization term for item biases
        
        user_bias_reg : (float)
            Regularization term for user biases
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.ratings = ratings
        self.nb_users, self.nb_items = ratings.shape
        self.K = K
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.nb_samples = len(self.sample_row)
        self._v = verbose


    def train(self, nb_iter=10, learning_rate=0.001):
        """ Train model for nb_iter iterations from scratch."""
        
        # Initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.K, size=(self.nb_users, self.K))
        self.item_vecs = np.random.normal(scale=1./self.K, size=(self.nb_items, self.K))
        
        self.learning_rate = learning_rate
        self.user_bias = np.zeros(self.nb_users)
        self.item_bias = np.zeros(self.nb_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
        self.partial_train(nb_iter)
    
    
    def partial_train(self, nb_iter):
        """ 
        Train model for nb_iter iterations. Can be 
        called multiple times for further training.
        """
        
        # Initialization
        ctr = 1
        
        while ctr <= nb_iter:
            if ctr % 10 == 0 and self._v:
                print('\tcurrent iteration: {}'.format(ctr))
                
            self.training_indices = np.arange(self.nb_samples)
            np.random.shuffle(self.training_indices)
            self.sgd()
            ctr += 1


    def sgd(self):
        """ Stochastic gradient descent algorithm."""        
        
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u, i] - prediction) # error
            
            # Update biases
            self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
            if np.isnan(self.user_bias[u]):
                self.user_bias[u] = 0.0
            self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])
            if np.isnan(self.item_bias[i]):
                self.item_bias[i] = 0.0
            
            #Update latent factors
            self.user_vecs[u, :] += self.learning_rate * (e * self.item_vecs[i, :] - self.user_fact_reg * self.user_vecs[u,:])
            self.item_vecs[i, :] += self.learning_rate * (e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i,:])
               
                      
    def predict(self, u, i):
        """ Single user and item prediction."""
        
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction
    
    
    def predict_all(self):
        """ Predict ratings for every user and item."""
        
        predictions = np.zeros((self.user_vecs.shape[0], self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    
    
    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of mae as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mae : (list)
            Training data mae values for each value of iter_array
        test_mae : (list)
            Test data mae values for each value of iter_array
        """
        
        # Initialization
        iter_array.sort()
        self.train_mae = []
        self.test_mae = []
        iter_diff = 0
        
        for (i, nb_iter) in enumerate(iter_array):
            if self._v:
                print('Iterations: {}'.format(nb_iter))
            if i == 0:
                self.train(nb_iter - iter_diff, learning_rate)
            else:
                self.partial_train(nb_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mae += [get_mae(predictions, self.ratings)]
            self.test_mae += [get_mae(predictions, test)]
            if self._v:
                print('Train mae: ' + str(self.train_mae[-1]))
                print('Test mae: ' + str(self.test_mae[-1]))
            iter_diff = nb_iter
           
           
# Main function
if __name__ == '__main__':
    
    # Parse the CSV file
    p = parser.parser()
    p.parse_csv()
    
    # Split into a train and a test arrays
    train, test = p.train_test_split()
    
    # Compute
    mf = mf(train, 5, verbose=True)
    iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
    mf.calculate_learning_curve(iter_array, test, learning_rate=0.001)
    
    # Plot
    plot_learning_curve(iter_array, mf)