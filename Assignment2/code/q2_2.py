'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        means[i] = np.mean([train_data[j] for j in range(len(train_data)) if train_labels[j] == i], axis=0)
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)

    # Compute covariances
    for i in range(10):
        data_class_i = np.squeeze([train_data[j,:] for j in range(len(train_data)) if train_labels[j] == i])
        data_size = len(data_class_i)

        left = np.transpose(data_class_i - means[i,:])
        right = data_class_i - means[i,:]
        covariances[i] = np.dot(left, right) / data_size

        # for j in range(64):
        #     for k in range(64):
        #         left = np.transpose( [(data_class_i[m,j] - means[i,j]) for m in range(len(data_class_i))] )
        #         right = [(data_class_i[m,k] - means[i,k]) for m in range(len(data_class_i))]
        #
        #         covariances[i,j,k] = np.dot(left, right) / data_size

    # A self-check function for covariances calculation
    # cov_verify = np.zeros((10, 64, 64))
    # for i in range(10):
    #     x = [train_data[j] for j in range(len(train_data)) if train_labels[j] == i]
    #     cov_verify[i] =  np.cov( np.transpose(x) )

    return covariances


def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_diag = np.zeros((10,8,8))
    for i in range(10):
        cov_diag[i] = np.log( np.diag(covariances[i]).reshape(-1,8) )
        # ...
    # Plot all means on same axis
    all_concat = np.concatenate(cov_diag, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    result = []
    for digit in digits:
        log_likelihood = []
        for i in range(10):
            left = ((2*np.pi) ** -(64/2)) * (np.linalg.det(covariances[i] + 0.01 * np.identity(64)) ** (-1/2))
            right = np.exp((-1/2) * np.linalg.multi_dot([
                            np.transpose(digit - means[i]),
                            np.linalg.inv(covariances[i] + 0.01 * np.identity(64)),
                            digit - means[i]
                            ]))

            log_likelihood.append(np.log(left * right))
        result.append(log_likelihood)

    return result

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    result = []
    generative_likelihoods = generative_likelihood(digits, means, covariances)
    # print("generative_likelihood size")
    # print(np.shape(generative_likelihoods))
    for gen in generative_likelihoods:
        den = np.log( sum(np.exp(gen_k) for gen_k in gen) * (1/10))# sum over all 10 class

        conditional_likelihood = []
        for gen_k in gen:
            conditional_likelihood.append( gen_k + np.log(1/10) - den)
        result.append(conditional_likelihood)

    return result


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # print("cond_likelihood size:")
    # print(np.shape(cond_likelihood))
    sum_likelihood = 0
    for i in range(len(digits)):
        sum_likelihood += cond_likelihood[i][int(labels[i])]
    # Compute as described above and return
    return sum_likelihood / len(digits)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class

    return a (n,1) matrix where n is size of data_set, with each col
    represents the predicted class
    '''
    result =[]
    cond_likelihoods = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    for cond in cond_likelihoods:
        result.append(cond.index(max(cond)))
    return result

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    plot_cov_diagonal(covariances)

    # Evaluation
    avg_cond_like_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_cond_like_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    result_train_data = classify_data(train_data, means, covariances)
    result_test_data = classify_data(test_data, means, covariances)

    accuracy_training = sum(1 for i in range(len(train_data)) if result_train_data[i] == train_labels[i]) / len(train_data)
    accuracy_testing = sum(1 for i in range(len(test_data)) if result_test_data[i] == test_labels[i]) / len(test_data)
    print("Average conditional likelihood on")
    print("training set: %s" %avg_cond_like_train)
    print("testing set: %s" %avg_cond_like_test)
    print("Classification Accuracy on")
    print("training set: %s" % accuracy_training)
    print("testing set: %s" % accuracy_testing)

if __name__ == '__main__':
    main()
