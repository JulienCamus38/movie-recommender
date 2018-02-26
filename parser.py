# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:50 2018

@author: camusj
"""

try:
    import sys
    import numpy as np
    import pandas as pd
    np.random.seed(0)
except ImportError:
    raise ImportError('sys, numpy and pandas modules need to be imported.')
    
def create_dict_users(ratings):
    """
    """
    
    # Initialization
    dict_users = dict()
    fname = './ml-latest-small/ratings.csv'
    
    # Try/catch if file exists or not
    try:
        f = open(fname, 'r')
    except IOError:
        print("Could not read file: " + fname)
        sys.exit()
    
    with f:
        names = ['userId', 'movieId', 'rating', 'timestamp']
        df = pd.read_csv(f, names=names, skiprows=1)
        
        for row in df.itertuples():
            key = int(row[0])
            if key in dict_users.keys():
                dict_users[key].append(row[1:3])
            else:
                dict_users[key] = [row[1:3]]
    
    return dict_users

def create_dict_indexes(ratings):
    """
    """
    
    # Initialization
    dict_indexes = dict()
    fname = './ml-latest-small/ratings.csv'
    key = 0
    
    # Try/catch if file exists or not
    try:
        f = open(fname, 'r')
    except IOError:
        print("Could not read file: " + fname)
        sys.exit()
    
    with f:
        names = ['userId', 'movieId', 'rating', 'timestamp']
        df = pd.read_csv(f, names=names, skiprows=1)
        
        for row in df.itertuples():
            value = int(row[1])
            if value not in dict_indexes.values():
                dict_indexes[key] = value
                ++key
                
    return dict_indexes
    
def get_length_dict_users(dict_users, i):
    return len(dict_users[i])

# Parse the csv file and create the array of ratings: [[userId, movieId, rating]]
def parse_csv(fname='./ml-latest-small/ratings.csv'):
    """
    Parse a csv file into an array of [userId, movieId, rating].
    
    Argument:
    - fname: path of the csv file to parse
    """
    
    # Try/catch if file exists or not
    try:
        f = open(fname, 'r')
    except IOError:
        print("Could not read file: " + fname)
        sys.exit()
    
    with f:
        names = ['userId', 'movieId', 'rating', 'timestamp']
        df = pd.read_csv(f, names=names, skiprows=1)
        
        # Compute the number of users and the number of movies
        nb_users = df.userId.unique().shape[0]
        nb_movies = df.movieId.max()
        
        # Initialize the ratings array
        ratings = np.zeros((nb_users, nb_movies))
        
        # Filling
        for row in df.itertuples():
            ratings[int(row[1])-1, int(row[2])-1] = float(row[3])
        
        # Compute the sparsity
        sparsity = float(len(ratings.nonzero()[0]))
        sparsity /= (ratings.shape[0] * ratings.shape[1])
        sparsity *= 100
        
        # Print information
        print('df')
        df.head()
        print('Number of users = ' + str(nb_users))
        print('Number of movies = ' + str(nb_movies))
        print('Sparsity = {:4.2f}%'.format(sparsity))
        print('ratings')
        print(ratings)
        
        return ratings
        
def train_test_split(ratings):
    """
    Split the ratings array in a train array and a test array.
    
    Argument:
    - ratings: Array containing the ratings parsed from the csv file.
    """
    
    # Initialization
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    
    # Filling
    for user in range(ratings.shape[0]):
        # TODO: modify size=10 by 70%/30%
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.0
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    
    return train, test