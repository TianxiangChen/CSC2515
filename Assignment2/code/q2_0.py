'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)

        # Compute mean of class i
        column = []
        column = np.mean(i_digits, axis = 0)
        means.append( np.reshape(column, (8,8)) )

    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    plt.imshow(all_concat, cmap='gray')
    # for i in range(len(means)):
    #     plt.subplot(1,10,i+1)
    #     plt.imshow(means[i], cmap='gray')
    # plt.suptitle('Mean for digit 0-9')
    plt.show()

if __name__ == '__main__':
    train_data, train_labels, test_data, teste_labels = data.load_all_data_from_zip('a2digits.zip', 'data')
    plot_means(train_data, train_labels)
