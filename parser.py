# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:12:50 2018

@author: camusj
"""

try:
    import sys
except ImportError:
    raise ImportError('sys module needs to be imported.')
    
try:
    import numpy as np
    np.random.seed(0)
except ImportError:
    raise ImportError('numpy module needs to be imported.')
    
try:
    import pandas as pd
except ImportError:
    raise ImportError('pandas module needs to be imported.')
    
try:
    import math
except ImportError:
    raise ImportError('math module needs to be imported.')
    
class parser():
    def __init__(self,
                 fname='./data/ml-latest-small/ratings.csv',
                 dict_users=dict(),
                 dict_indexes=dict(),
                 nb_users=0,
                 nb_items=0,
                 ratings=np.zeros((0, 0)),
                 sparsity=0.0):
        """
        Parse a csv file and fill some useful objects  
        
        Arguments
        =========
        - fname : (string)
            Path of the csv file to parse        
        
        - dict_users : (dict)
            Dictionary representing {userId: (movieId, rating), ...}
    
        - dict_indexes : (dict)
            Dictionary representing {movieId: new_movieId}
            where new_movieId start from 0 and is incremented 1 by 1.

        - nb_users : (int)
            Number of users

        - nb_items : (int)
            Number of items (movies)
        
        - ratings : (ndarray)
            Array containing the ratings parsed from the csv file
            
        - sparsity : (float)
            Sparsity of the array, represents the number of 0 in the 
            ratings matrix
        """
        
        self.fname = fname
        self.dict_users = dict_users
        self.dict_indexes = dict_indexes
        self.nb_users = nb_users
        self.nb_items = nb_items
        self.ratings = ratings
        self.sparsity = sparsity
        
    def parse_csv(self):
        """
        Parse a csv file into an array of [userId, movieId, rating]
        
        Output
        ======
        - Fill dict_users with {userId: movieId}
        - Fill dict_indexes with {newMovieIdFromZero: movieId}
        """
        
        # Try/catch if file exists or not
        try:
            f = open(self.fname, 'r')
        except IOError:
            print("Could not read file: " + self.fname)
            sys.exit()
        
        with f:
            names = ['userId', 'movieId', 'rating', 'timestamp']
            df = pd.read_csv(f, names=names, skiprows=1)
            
            # Compute the number of users and the number of movies
            self.nb_users = df.userId.unique().shape[0]
            self.nb_items = df.movieId.unique().shape[0]
            
            # Initialization
            self.ratings = np.zeros((self.nb_users, self.nb_items))
            new_movieId = 0
            
            # Filling
            for row in df.itertuples():
                
                # dict_indexes
                movieId = int(row[2])
                if movieId not in self.dict_indexes.keys():
                    self.dict_indexes[movieId] = new_movieId
                    new_movieId += 1
                    
                # dict_users
                userId = int(row[1])
                if userId in self.dict_users.keys():
                    self.dict_users[userId].append(row[2:4])
                else:
                    self.dict_users[userId] = [row[2:4]]
                
                # ratings
                self.ratings[userId-1, self.dict_indexes[movieId]] = float(row[3])
            
            # Compute the sparsity
            self.sparsity = float(len(self.ratings.nonzero()[0]))
            self.sparsity /= (self.ratings.shape[0] * self.ratings.shape[1])
            self.sparsity *= 100
            
            # Print information
            sep = '================================================\n'
            print(sep + 'df.head()')
            print(df.head())
            print(sep + 'Number of users = ' + str(self.nb_users))
            print(sep + 'Number of movies = ' + str(self.nb_items))
            print(sep + 'Sparsity = {:4.2f}%'.format(self.sparsity))
            print(sep + 'ratings')
            print(self.ratings)
        
    def get_length_dict_users(self, user):
        """
        Get the number of existing ratings by a user
        
        Arguments
        =========
        - user : (int)
            User for which the ratings are counted
        """
        
        return len(self.dict_users[user])
            
    def train_test_split(self):
        """ Split the ratings array in a train array and a test array """
        
        # Initialization
        test = np.zeros(self.ratings.shape)
        train = self.ratings.copy()
        
        # Filling
        for user in range(1, self.ratings.shape[0]+1):
            # Length of userId subarray
            length_user_dict = self.get_length_dict_users(user)
            
            # Number of values in train and test subarrays (70%/30%)
            nb_user_test_values = math.floor(0.3*length_user_dict)
            
            test_ratings = np.random.choice(self.ratings[user-1, :].nonzero()[0], 
                                            size=nb_user_test_values,
                                            replace=False)
            train[user-1, test_ratings] = 0.0
            test[user-1, test_ratings] = self.ratings[user-1, test_ratings]
            
        # Test and training are truly disjoint
        assert(np.all((train * test) == 0))
        
        return train, test