'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from time import time
from scipy.stats import randint as sp_randint
from sklearn.svm import SVC

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

if __name__ == '__main__':
    train_data, test_data = load_data()
    # train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_bow, test_bow, feature_names = tf_idf_features(train_data, test_data)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)


    #----------------------------- K-NN ----------------------------------------
    # neigh = KNeighborsClassifier(n_neighbors=5)
    # neigh.fit(train_bow, train_data.target)
    # print("K-NN (k=5) accuracy for training set: %s" %(neigh.score(train_bow,train_data.target)))
    # print("K-NN (k=5) accuracy for testing set: %s" %(neigh.score(test_bow,test_data.target)))


    #----------------------------- R-NN ----------------------------------------
    # r_neigh = RadiusNeighborsClassifier(radius=3.0)
    # r_neigh.fit(train_bow, train_data.target)
    # print("R-NN (r=1) accuracy for training set: %s" %(r_neigh.score(train_bow,train_data.target)))
    # print("R-NN (r=1) accuracy for testing set: %s" %(r_neigh.score(test_bow,test_data.target)))


    #----------------------------- RandomCV ------------------------------------
    # build a classifier
    # clf = RandomForestClassifier(n_estimators=20)
    #
    # # specify parameters and distributions to sample from
    # param_dist = {"max_depth": [3, None],
    #           "max_features": sp_randint(1, 11),
    #           "min_samples_split": sp_randint(2, 11),
    #           "min_samples_leaf": sp_randint(1, 11),
    #           "bootstrap": [True, False],
    #           "criterion": ["gini", "entropy"]}
    # # run randomized search
    # n_iter_search = 20
    # random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
    #                                n_iter=n_iter_search)
    #
    # start = time()
    # random_search.fit(train_bow, train_data.target)
    # print("RandomizedSearchCV took %.2f seconds for %d candidates"
    #     " parameter settings." % ((time() - start), n_iter_search))
    # # report(random_search.cv_results_)
    # print("RandomCV accuracy for training set: %s" %(random_search.score(train_bow,train_data.target)))
    # print("RandomCV accuracy for testing set: %s" %(random_search.score(test_bow,test_data.target)))

    #----------------------------- SVM -----------------------------------------
    svm_clf = SVC(kernel='linear', decision_function_shape='ovo')
    svm_clf.fit(train_bow, train_data.target)
    print("SVM accuracy for training set: %s" %(svm_clf.score(train_bow,train_data.target)))
    print("SVM accuracy for testing set: %s" %(svm_clf.score(test_bow,test_data.target)))


    #----------------------------- Decision Tree -------------------------------
    # d_tree = DecisionTreeClassifier()
    # d_tree.fit(train_bow, train_data.target)
    # print("Decision Tree accuracy for training set: %s" %(d_tree.score(train_bow,train_data.target)))
    # print("Decision Tree accuracy for testing set: %s" %(d_tree.score(test_bow,test_data.target)))

    #----------------------------- Logistic Regression -------------------------
    # logreg = linear_model.LogisticRegression(C=1e5)
    # we create an instance of Neighbours Classifier and fit the data.
    # logreg.fit(train_bow, train_data.target)
    # print("Logistic Regression accuracy for training set: %s" %(logreg.score(train_bow,train_data.target)))
    # print("Logistic Regression accuracy for testing set: %s" %(logreg.score(test_bow,test_data.target)))
