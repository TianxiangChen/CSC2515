'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    alpha = 2
    beta = 2

    for i in range(10):
        for j in range(64):
            N_c = sum(train_data[k][j] for k in range(len(train_data)) if train_labels[k]==i)
            N = sum(1 for k in range(len(train_data)) if train_labels[k]==i)
            eta[i][j] = (N_c + alpha - 1) / (N + alpha + beta - 2)
    return eta


def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''

    all_concat = np.concatenate([i.reshape(-1,8) for i in class_images], 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(10):
        for j in range(64):
            generated_data[i][j] = np.random.binomial(1,eta[i][j])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    result = []
    for digit in bin_digits:
        generative_likelihood =[]
        for i in range(10):
            like = 1
            for j in range(64):
                like *= (eta[i][j] ** digit[j]) * ((1-eta[i][j]) ** digit[j])
            generative_likelihood.append(np.log(like))
        result.append(generative_likelihood)

    return result


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    result = []
    generative_likelihoods = generative_likelihood(bin_digits, eta)

    for gen in generative_likelihoods:
        den = np.log( sum(np.exp(gen_k) for gen_k in gen) * (1/10))# sum over all 10 class
        conditional_likelihood = []
        for gen_k in gen:
            conditional_likelihood.append( gen_k + np.log(1/10) - den)
        result.append(conditional_likelihood)

    return result


def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    sum_likelihood = 0
    for i in range(len(bin_digits)):
        sum_likelihood += cond_likelihood[i][int(labels[i])]
    # Compute as described above and return
    return sum_likelihood / len(bin_digits)


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    return [cond.index(max(cond)) for cond in cond_likelihood]


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)

    result_train_data = classify_data(train_data, eta)
    result_test_data = classify_data(test_data, eta)
    avg_cond_like_train = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_cond_like_test = avg_conditional_likelihood(test_data, test_labels, eta)
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
