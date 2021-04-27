import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from pandas import read_csv
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# constant values

# set the percentage of of data to split into 30% test
# data and 70% training data as this is a commonly used ratio
TEST_SIZE = 0.3

# set the random state of the classifier to one to ensure
# repeatability by using a constant value
RANDOM_STATE = 1

# set the number of cross validation folds to 10 as this
# is a commonly used value
CV_FOLDS = 10

# set the tolerance of the neural network classifier to 1e-2
# so the maximum amount of iterations is never reached
TOLERANCE = 1e-2

# set the values of the classification of a malignant or benign tumour
MALIGNANT = 1
BENIGN = 0

# set values to be used for indexing the data by columns using numpy
FIRST_COLUMN = 1
SECOND_COLUMN = 2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def prepare_dataset(dataset_path):
    """
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    """
    # load in the data as a numpy array and set
    # the header parameter to None as column names are
    # not expected in the dataset
    data = np.array(read_csv(dataset_path, header=None))

    # set the values of the X and y by indexing the numpy
    # array 'data' so that y is the classification of the data
    # (malignant or benign) and X is the rest of the dataset,
    # then set them to be numpy arrays
    X, y = np.array(data[:, SECOND_COLUMN:]), np.array([MALIGNANT if index == 'M'
                                                        else BENIGN for index in data[:, FIRST_COLUMN]])

    # uncomment to view the values of X and y
    # print(X)
    # print(y)

    # return the X and y numpy arrays
    return X, y


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    """
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    """
    # set possible values for the parameter max_depth
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]

    # set the parameters that will be used in cross validation
    params = {"max_depth": max_depths}

    # setup cross validation using a Decision Tree classifer,
    # with the parameters set earlier, random_state set to 1 
    # for repeatability and cv set to 10 folds
    cvSearch = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), params, cv=CV_FOLDS)

    # fit the data to the cross validated classifier
    clf = cvSearch.fit(X_training, y_training)

    # uncomment to view the best parameters found via
    # cross validation
    # print(clf.best_params_)

    # return the classifier
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    """
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
    X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

	@return
	clf : the classifier built in this function
	"""
    # set possible values for the parameter n_neighbors
    n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # set the parameters that will be used in cross validation
    params = {"n_neighbors": n_neighbors}

    # setup cross validation using a Nearest Neighbors classifer,
    # with the parameters set earlier, random_state set to 1 
    # for repeatability and cv set to 10 folds
    cvSearch = GridSearchCV(KNeighborsClassifier(), params, cv=CV_FOLDS)

    # fit the data to the cross validated classifier
    clf = cvSearch.fit(X_training, y_training)

    # uncomment to view the best parameters found via
    # cross validation
    # print(clf.best_params_)

    # return the classifier
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    """
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    """
    # set possible values for the parameter C
    Cs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # set the parameters that will be used in cross validation
    params = {"C": Cs}

    # setup cross validation using a Support Vector Machine classifer,
    # with the parameters set earlier, random_state set to 1 
    # for repeatability and cv set to 10 folds
    cvSearch = GridSearchCV(SVC(random_state=RANDOM_STATE), params, cv=CV_FOLDS)

    # fit the data to the cross validated classifier
    clf = cvSearch.fit(X_training, y_training)

    # uncomment to view the best parameters found via
    # cross validation
    # print(clf.best_params_)

    # return the classifier
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    """
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    """
    # set possible values for the parameter hidden_layer_sizes
    hidden_layer_sizes = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]

    # set the parameters that will be used in cross validation
    params = {"hidden_layer_sizes": hidden_layer_sizes}

    # setup cross validation using a Support Vector Machine classifer,
    # with the parameters set earlier, random_state set to 1 
    # for repeatability, tol set to 1e-2 so that the maximum iteration is
    # never reached and cv set to 10 folds
    cvSearch = GridSearchCV(MLPClassifier(tol=TOLERANCE, random_state=RANDOM_STATE), params, cv=CV_FOLDS)

    # fit the data to the cross validated classifier
    clf = cvSearch.fit(X_training, y_training)

    # uncomment to view the best parameters found via
    # cross validation
    # print(clf.best_params_)

    # return the classifier
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    # set the path of the medical data and prepare it into usable numpy arrays
    path = './medical_records.data'
    X, y = prepare_dataset(path)

    # setup training and testing data for X and y using train_test_split
    # with random_state set to 1 for repeatability and using the test_size specified earlier
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE)

    # group the classifiers into a list that can be iterated over
    clfs = [build_DecisionTree_classifier,
            build_NearrestNeighbours_classifier,
            build_SupportVectorMachine_classifier,
            build_NeuralNetwork_classifier]

    # for each classifier in the list clfs, run sklearn report on 
    # classifier predictions using training and test data
    for func in clfs:
        # start timing the computation of the classifier
        startTime = time.time()

        # prepare the classifier
        clf = func(X_train, y_train)

        # run report on the training set prediction
        print(func.__name__, ": TRAINING")
        print(classification_report(y_train, clf.predict(X_train)))

        # run report on the test set prediction
        print(func.__name__, ": TEST")
        print(classification_report(y_test, clf.predict(X_test)))

        # end the timer of the classifier
        endTime = time.time()

        # print the time taken for the classifier to compute
        print(func.__name__, 'computation time:', endTime - startTime, 'seconds')
