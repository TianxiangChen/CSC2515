'''
Question 1 Skeleton Code
'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
    # train_pred = model.predict(binary_train)
    # print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    # test_pred = model.predict(binary_test)
    # print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def prediction_accuracy(predictions, targets):
    return (100.0/len(predictions)) * sum([1 if predictions[i] == targets[i] else 0 for i in range(len(predictions))])


def compute_confusion_matrix(test_predictions, test_targets):
    matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            for n in range(len(test_predictions)):
                if test_predictions[n] == i and test_targets[n] == j:
                    matrix[i][j] += 1
    return matrix


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names_bow = bow_features(train_data, test_data)
    tf_idf_train, tf_idf_test, tf_idf_feature_names = tf_idf_features(train_data, test_data)

    def try_all_models(train, train_target, test, test_target, feature_type_name):
        print(feature_type_name + " ---------------")

        def run_model(model):
            print("Train accuracy: " + str(prediction_accuracy(model.predict(train), train_target)))
            print("Test accuracy: " + str(prediction_accuracy(model.predict(test), test_target)))

        bnb_model = bnb_baseline(train, train_target, test, test_target)
        print("BNB Baseline Model")
        run_model(bnb_model)

        sgd_classifier = SGDClassifier(max_iter=50)
        sgd_classifier.fit(train, train_target)
        print("SGDClassifier")
        run_model(sgd_classifier)

        linear_svc = LinearSVC()
        linear_svc.fit(train, train_target)
        print("Linear SVC")
        run_model(linear_svc)

        logistic_regr = LogisticRegression()
        logistic_regr.fit(train, train_target)
        print("Logistic Regression")
        run_model(logistic_regr)

        # knn = KNeighborsClassifier(n_neighbors=3)
        # knn.fit(train, train_target)
        # print("KNN")
        # run_model(knn)

        # random_forest = RandomForestClassifier()
        # random_forest.fit(train, train_target)
        # print("Random Forest")
        # run_model(random_forest)

        # logistic_regr_m = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        # logistic_regr_m.fit(train, train_target)
        # print("Logistic Regression, multinomial")
        # run_model(logistic_regr_m)

    # try_all_models(train_bow, train_data.target, test_bow, test_data.target, "BOW")
    try_all_models(tf_idf_train, train_data.target, tf_idf_test, test_data.target, "Tf-idf")

    sgd_classifier = SGDClassifier(max_iter=50)
    sgd_classifier.fit(tf_idf_train, train_data.target)
    sgd_classifier_predictions = sgd_classifier.predict(tf_idf_test)
    matrix = compute_confusion_matrix(sgd_classifier_predictions, test_data.target)
    plt.imshow(matrix, cmap='gray')
    plt.colorbar()
    print("Done...")
