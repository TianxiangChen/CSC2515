import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")
BATCHES = 50
eta = 2e-6 #learning rate, which i used for finding w (the related code is commented out)
Round = 500

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) #add constant one feature - no bias needed
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w, m): #Add m here for batch size
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    batch_grad = -2 * np.dot(X.transpose() , (y - np.dot(X,w))) / m

    return batch_grad
    # raise NotImplementedError()

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    w_backup = w #backup the w, for true/mini-batch gradient comparison

    # Traditional way to calculate gradient
    batch_grad_old = np.zeros(w.shape[0])
    batch_grad_old = lin_reg_gradient(X, y, w, len(X))
    print 'True gradient: \n', batch_grad_old


    # Use mini-batch to calculate gradient
    batch_grad_new = np.zeros(w.shape[0])
    for i in range(Round):
        # Initialize w again, using the same w as true gradient
        w = w_backup
        X_b, y_b = batch_sampler.get_batch()
        batch_grad_new += lin_reg_gradient(X_b, y_b, w, BATCHES)/Round
    print 'Gradient using mini-batch:\n', batch_grad_new

    batch_diff = np.zeros(w.shape[0])
    print 'the gradient diff in square distance is %.1f' %(np.sum((batch_grad_new - batch_grad_old)**2))
    print 'the gradient diff using cos similiarity is %.6f' %(cosine_similarity(batch_grad_new, batch_grad_old))


    # Code for Calculating the w using gradient and verify with target y
    # # Initialize w again
    # w = np.random.randn(w.shape[0])
    # batch_grad = np.zeros(w.shape[0])
    # error = 1 # Defined for the sum for squared difference between y_hat and y
    # w_temp = np.zeros(w.shape[0])
    # while error >= 2e-14:
    #     batch_grad = -0.5 * np.dot(X.transpose() , (y - np.dot(X,w))) / len(X)
    #     w_temp = w
    #     w = w - eta * batch_grad
    #     error = (np.sum((w-w_temp)**2))
    #     print 'error = %.4g, squared dist error = %.4f, cos similiarity = %.4f' %(error, np.sum(y - np.dot(X,w))**2, cosine_similarity(y, np.dot(X, w)))
    # y_pred = np.dot(X, w)
    # mse = ((y - y_pred)**2).mean()
    # print 'Final result after error goes <= 2e-14'
    # print 'squared dist error = %.4f, cos similiarity = %.4f' %(np.sum(y - np.dot(X,w))**2, cosine_similarity(y, np.dot(X, w)))
    # print 'mse = %.4f' %(mse)
    ############################################

    print 'it takes several seconds for q3.6 plot'
    # Initialize w again
    w = np.random.randn(w.shape[0])
    w_backup = w
    m = np.linspace(1,400,400)
    var = np.zeros(len(m))
    for i in range(len(m)):
        batch_grad_last = np.zeros(Round)
        for j in range(Round):
            w = w_backup
            X_b, y_b = batch_sampler.get_batch()
            # I randomly choose w5 for wj in this question (random decision by me...)
            batch_grad_last[j] = lin_reg_gradient(X_b, y_b, w, i)[5]
        avg = np.mean(batch_grad_last)

        for k in range(Round):
            var[i] += ((batch_grad_last[k] - avg)**2) / Round
    plt.plot(m,var)
    plt.title(r'log($\sigma$) vs. log(m)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
