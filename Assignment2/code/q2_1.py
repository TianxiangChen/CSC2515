'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
# Define as 10-cross validation
k_cross = 10



class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        # train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''

        while(1):
            dist = self.l2_distance(test_point)
            index = dist.argsort()[:k]

            vote = [0 for x in range(10)]
            for i in range(k):
                major_class = int(np.squeeze(self.train_labels[index[i]]))
                vote[major_class] +=1
            m = max(vote)
            index_of_max = [i for i, j in enumerate(vote) if j == m]

            if len(index_of_max) == 1:
                digit = index_of_max[0]
                break
            else:
                k -=1

        return digit


def cross_validation(knn, k_range=np.arange(1,16)): # change to 16 since we want k=1-15
    size_data = len(knn.train_labels)
    subset_size = int(size_data / k_cross)

    # Generate shuffled indices into dataset
    random_indices = np.random.permutation(range(size_data))
    accuracy = np.zeros((len(k_range), k_cross))


    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        for i in range(k_cross):
            # print('The program is testing for k = %d. It finishes %d/%d' %(k, i+1, k_cross))
            validation_indices = random_indices[i*subset_size: (i+1)*subset_size]
            # Remove validation set from the whole set to form training set
            train_indices = [x for x in random_indices if x not in validation_indices]
            knn_subtrain = KNearestNeighbor(knn.train_data[train_indices],
                knn.train_labels[train_indices])
            accuracy[k-1,i] = classification_accuracy(knn_subtrain, k,
                knn.train_data[validation_indices], knn.train_labels[validation_indices])
    accuracy = np.mean(accuracy,axis=1)
    return accuracy


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    counter_all = 0
    counter_correct = 0
    for i in range(len(eval_data)):
        if knn.query_knn(eval_data[i], k) == eval_labels[i]:
            counter_correct += 1
        counter_all += 1
    return (counter_correct / counter_all)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # k = 10
    # print(knn.query_knn(test_data[0], k))
    # print(test_labels[0])

    k = 1
    accuracy_train = classification_accuracy(knn, k, train_data, train_labels)
    accuracy_test = classification_accuracy(knn, k, test_data, test_labels)
    print("accuracy for k=%d\ntraining set: %.3f\ntesting set: %.3f" %(k, accuracy_train, accuracy_test))

    k = 15
    accuracy_train = classification_accuracy(knn, k, train_data, train_labels)
    accuracy_test = classification_accuracy(knn, k, test_data, test_labels)
    print("accuracy for k=%d\ntraining set: %.3f\ntesting set: %.3f" %(k, accuracy_train, accuracy_test))

    print("Perform the %d-cross validation for finding optimal k: " %k_cross)
    accuracy = cross_validation(knn)
    for i in range(len(accuracy)):
        print("%dNN: %.3f%%" %(i+1, accuracy[i]*100))

    m = max(accuracy)
    # Very rare but still check if there is a tier
    index_of_max = [i for i, j in enumerate(accuracy) if j == m]

    print("The best result for kNN is with k = ", end ='')
    for i in range(len(index_of_max)):
        if i == 0:
            print(index_of_max[i]+1, end ='')
        else:
            print(", %d" %index_of_max[i]+1, end='')
    print("")

    for i in range(len(index_of_max)):
        accuracy_train = classification_accuracy(knn, index_of_max[i]+1, train_data, train_labels)
        accuracy_test = classification_accuracy(knn, index_of_max[i]+1, test_data, test_labels)
        print("Accuracy for optimal kNN k=%d\ntraining set: %.3f\ntesting set: %.3f" %(index_of_max[i]+1, accuracy_train, accuracy_test))


if __name__ == '__main__':
    main()
