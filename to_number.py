#!/usr/bin/python

from __future__ import division
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import math
import sys
import ast
import csv
import operator


# convert categorical data to number so that data can be used by sklearn.

def read_data():
    training_file_path = "/home/ubuntu/PycharmProjects/DecisionTree/adult.data.csv"
    testing_file_path = "/home/ubuntu/PycharmProjects/DecisionTree/adult.test.csv"
    print "Reading training data..."
    training_file = pd.read_csv(training_file_path)
    testing_file = pd.read_csv(testing_file_path)
    # print training_file.head()
    # print testing_file.dtypes

    return [training_file, testing_file]


def preprocess(training_file, testing_file):
    print "Preprocessing data..."
    # processed_training_file = pd.get_dummies(training_file)
    # print processed_training_file.head()

    le = LabelEncoder()
    for col in testing_file.columns.values:
        # Encoding only categorical variables
        if testing_file[col].dtypes == 'object':
        # Using whole data to form an exhaustive list of levels
            data = training_file[col].append(testing_file[col])
            le.fit(data.values)
            training_file[col] = le.transform(training_file[col])
            testing_file[col] = le.transform(testing_file[col])
    
    training_y = training_file['salary']
    training_y = pd.DataFrame({'salary':training_y.values})
    training_X = training_file.drop('salary', axis=1)
    testing_y = testing_file['salary']
    testing_y = pd.DataFrame({'salary':testing_y.values})
    testing_X = testing_file.drop('salary', axis=1)


    print training_y.head(10)

    training_X.to_csv("./training_X.csv", mode='w', index=False)
    training_y.to_csv("./training_y.csv", mode='w', index=False)
    testing_X.to_csv("./testing_X.csv", mode='w', index=False)
    testing_y.to_csv("./testing_y.csv", mode='w', index=False)

    return [training_X, training_y, testing_X, testing_y]


def main():
    training_file, testing_file = read_data()
    training_X, training_y, testing_X, testing_y = preprocess(training_file, testing_file)


if __name__ == "__main__":
    main()
