#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:50:26 2018

@author: nate
"""

def is_working():
    print("classify.py is working!")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:32:15 2017

@author: Nate Currit
"""

#import numpy as np
#import geopandas as gpd
from scipy.stats import expon
from sklearn import svm
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.naive_bayes import GaussianNB
#import datetime
import pickle
#import matplotlib.pyplot as plt
#from pprint import pprint


#def assign_actual(segments_path, training_path):
#    """
#    Assign land cover classes to training data.
#    
#    From a geospatial (polygon) vector file of image segments and their
#    attributes, do 3 things:
#        (1) Add a field named 'random' and assign a random value between
#            0 and 1 to each record (segment).
#        (2) Add a field named 'actual' that will store the integer
#            representation of the actual land cover class for each segment.
#        (3) Select segments that are completely within manually digitized
#            land cover polygons used to define training data, and... 
#        (4) Assign the appropriate land cover (integer) code for each segment
#            in the 'actual' field. 
#
#    Parameters
#    ----------
#    segments_path: string
#        The string is the path to an ESRI Shapefile, read with GeoPandas
#        into a GeoDataFrame. This Shapefile contains the image segments. 
#
#    training_path: string
#        The string is the path to an ESRI Shapefile, read with GeoPandas
#        into a GeoDataFrame. This Shapefile contains the manually digitized
#        training data.
#
#    Returns
#    -------
#    gdf: GeoPandas GeoDataFrame
#        A GeoDataFrame with the attributes for classification and the 'random'
#        and 'actual' fields added.
#
#    """
#    pass


def train_classifier(segments, actual, output_filename):
    """
    Train classification algorithm.
    
    Train the Support Vector Machine classification algorithm using the
    specified fields. 

    Parameters
    ----------
    segments: numpy 2D array
        A 2D numpy array where there is one row for each segment and each
        column represents an attribute of the segments. 

    actual: numpy 1D array
        A 1D numpy array equal in length to the number of records in segments.
        The single column contains actual class values for each of the
        segments.

    output_filename: string
        Output filename of the pickled trained SVM model.

    Returns
    -------
    model: svm.SVC
        Returns a trained SVM model that can be used to classify other data.

    """
    clf = svm.SVC()
        
    # specify parameters and distributions to sample from
    param_dist = {'C': expon(scale=100),
                  'gamma': expon(scale=.1),
                  'kernel': ['rbf'],
                  'class_weight':['balanced', None]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

    random_search.fit(segments, actual) # this may take time...
    pickle.dump(random_search, open(output_filename, "wb"))
    
    return clf


def predict(model, segments):
    """
    Classify segments using a trained SVM model

    Classify image segments using the trained Support Vector Machine model. 

    Parameters
    ----------
     model: svm.SVC
        A trained SVM model that can be used to classify other data.

    segments: numpy 2D array
        A 2D numpy array where there is one row for each segment and each
        column represents an attribute of the segments. Identical to segments
        from the train_classifier function.
    """
    predictions = model.predict(segments)

    return predictions


def main():
    pass
#    src = gpd.read_file('./segs/training_segments.shp')
#    
#    fields = ['count', 'perimeter', 'eccentrici', 'equal_diam', 'major_axis',
#              'minor_axis', 'orientatio', 'red_mean', 'green_mean',
#              'blue_mean', 'sobel_mean', 'sobel_sum', 'sobel_std']
#    
##    svm_t = train_classifier(src, 'trained_svm_' +
##                             str(datetime.datetime.now()).replace(" ", "_"),
##                             fields, 'class_id')
##    pprint(vars(svm_t))
#    
#    new_model = pickle.load(open("trained_svm_2018-03-19_21:51:24.707615", "rb"))
#    pprint(vars(new_model))
#
#    random_pct = 0.7
#    verification = src.loc[(src.class_id != 0) &
#                            (src.random < random_pct), fields]
#    verification_class = src.loc[(src.class_id != 0) & (src.random < random_pct),
#                             ['class_id']]
#    
#    X = verification.values
#    Y = verification_class.values.reshape(-1)
#
#    predictions = predict(new_model, X, fields)

##    crosstab = cross_tabulation(Y, predictions)
##    print(crosstab)


if __name__ == "__main__":
    main()
