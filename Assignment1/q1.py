from sklearn import datasets
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(1,figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
	#TODO: Plot feature i against y
        plt.plot(X[:,i],y,'.')
	plt.xlabel(features[i])
	plt.ylabel('Price')

    plt.tight_layout()
    #plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Add bias term
    col_ones = np.ones(len(X))
    X=np.column_stack((col_ones,X))
    # Remember to use np.linalg.solve instead of inverting!
    w_opt = np.linalg.solve(np.dot(X.transpose(), X), np.dot(X.transpose(), Y))
    return w_opt


def main():
    # Load the data
    X, y, features = load_data()
    print "Input data size: ", X.shape," Each input has ", X.shape[1], " features, as shown below."
    print "Features:\n {}".format(features)
    print "Target size: ",len(y)

    # Visualize the features
    visualize(X, y, features)

    #Normalize the data(X), for later comparing w
    for i in range(X.shape[1]):
        col_max = X[:,i].max()
        X[:,i] /= col_max

    #TODO: Split data into train and test
    i_train = np.random.choice(len(X), int(len(X)*0.8), replace = False)
    i_train = sorted(i_train)
    i = range(len(X))
    i_test = list(set(i) - set(i_train))

    # Separate into two sets
    X_train = X[i_train,:]
    y_train = y[i_train]
    X_test = X[i_test,:]
    y_test = y[i_test]

    # Fit regression model
    w_opt = fit_regression(X_train, y_train)
    col_ones = np.ones(len(X_test))
    X_test = np.column_stack((col_ones,X_test))
    y_pred = np.dot(X_test, w_opt)

    plt.figure(2)
    plt.scatter(y_test,y_pred)
    plotlim = int(max(max(y_test),max(y_pred)) *1.1)
    plt.plot([0,plotlim],[0,plotlim],'r')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

    features = np.insert(features,0,'Bias') # adding bias to features names array
    for i in range(len(features)):
        print "{:8s}| {:6f}".format(features[i],w_opt[i])
    print ""
    # Compute fitted values, MSE, etc.
    mse = ((y_test - y_pred)**2).mean()
    print 'The mean square error is ', mse

    rms = sqrt(mse)
    print 'The root mean square error is ', rms

    # Mean Absolute Error
    mae = (np.absolute(y_test - y_pred)).mean()
    print 'Mean absolute error is ', mae

    # plot all at the end
    plt.show()

if __name__ == "__main__":
    main()
