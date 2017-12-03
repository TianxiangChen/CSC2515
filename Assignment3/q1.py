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
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

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

def knn_model(train_input, train_target, test_input, test_target):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_input, train_target)
    print("K-NN (k=5) accuracy for training set: %s" %(neigh.score(train_input, train_target)))
    print("K-NN (k=5) accuracy for testing set: %s" %(neigh.score(test_input,test_target)))

def rnn_model(train_input, train_target, test_input, test_target):
    r_neigh = RadiusNeighborsClassifier(radius=3.0)
    r_neigh.fit(train_input, train_target)
    print("R-NN (r=1) accuracy for training set: %s" %(r_neigh.score(train_input,train_target)))
    print("R-NN (r=1) accuracy for testing set: %s" %(r_neigh.score(test_input,test_target)))

def svm_linear_model(train_input, train_target, test_input, test_target):
    # For this data set, we have more than 100,000 input feature but only ~10,000 inputs
    # To avoid overfitting, we cannot use complicated kernel but just linear one.
    svm_clf = LinearSVC()
    svm_clf.fit(train_input, train_target)
    print("SVM accuracy for training set: %s" %(svm_clf.score(train_input,train_target)))
    print("SVM accuracy for testing set: %s" %(svm_clf.score(test_input,test_target)))

def decision_tree_model(train_input, train_target, test_input, test_target):
    d_tree = DecisionTreeClassifier()
    d_tree.fit(train_input, train_target)
    print("Decision Tree accuracy for training set: %s" %(d_tree.score(train_input,train_target)))
    print("Decision Tree accuracy for testing set: %s" %(d_tree.score(test_input,test_target)))

def logistic_regression_model(train_input, train_target, test_input, test_target):
    logreg = LogisticRegression(C=1e7)
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(train_bow, train_data.target)
    print("Logistic Regression accuracy for training set: %s" %(logreg.score(train_bow,train_data.target)))
    print("Logistic Regression accuracy for testing set: %s" %(logreg.score(test_bow,test_data.target)))

def neural_network_model(train_input, train_target, test_input, test_target):
    clf = MLPClassifier(hidden_layer_sizes=(100,))
    clf.fit(train_input, train_target)
    print("Neural network accuracy for training set: %s" %(clf.score(train_input,train_target)))
    print("Neural network accuracy for testing set: %s" %(clf.score(test_input,test_target)))

def multinomialNB_model(train_input, train_target, test_input, test_target):
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_input, train_target)
    print("MultinomialNB accuracy for training set: %s" %(clf.score(train_input,train_target)))
    print("MultinomialNB accuracy for testing set: %s" %(clf.score(test_input,test_target)))

def compute_confusion_matrix(test_prediction, test_target):
    matrix = np.zeros((20, 20))

    for i in range(len(test_prediction)):
        matrix[test_prediction[i]][test_target[i]] += 1

    return matrix

def find_confusion_class(matrix):
    # this function finds the index of largest element in matrix which is not
    # in the diagonal line
    max_temp = 0
    matrix = matrix + matrix.transpose()
    for i in range(20):
        for j in range(20):
            if i != j:
                if matrix[i][j] > max_temp:
                    max_temp = matrix[i][j]
                    index_i = i
                    index_j = j
    # print("the largest Cij not in the diagonal is C[%d][%d]" %(index_i, index_j))
    return index_i, index_j

if __name__ == '__main__':
    train_data, test_data = load_data()
    # train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_bow, test_bow, feature_names = tf_idf_features(train_data, test_data)
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)




    # knn_model(train_bow, train_data.target, test_bow, test_data.target)
    # rnn_model(train_bow, train_data.target, test_bow, test_data.target)
    # svm_linear_model(train_bow, train_data.target, test_bow, test_data.target)
    # decision_tree_model(train_bow, train_data.target, test_bow, test_data.target)
    # logistic_regression_model(train_bow, train_data.target, test_bow, test_data.target)
    # multinomialNB_model(train_bow, train_data.target, test_bow, test_data.target)
    neural_network_model(train_bow, train_data.target, test_bow, test_data.target)


    # logreg = LogisticRegression(C=1e7)
    # logreg.fit(train_bow, train_data.target)
    # matrix = compute_confusion_matrix(logreg.predict(test_bow),test_data.target)
    # plt.imshow(matrix, cmap='gray_r')
    # plt.colorbar()
    # plt.show()
    # i,j = find_confusion_class(matrix)
    # print ("the two most confusion class are :\nclass %s: %s\nclass %s: %s" %(i, train_data.target_names[i], j, train_data.target_names[j]))
