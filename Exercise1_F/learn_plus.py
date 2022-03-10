# used for manipulating directory paths
from cProfile import label
import os
from traceback import print_tb

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import markers, projections, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# library written for this exercise providing additional functions for assignment submission, and others
import utils 

# define the submission/grader object for this exercise
grader = utils.Grader()

# Load data
data = np.loadtxt(os.path.join('Exercise1/Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
    
def featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.#!!!!!!!!!!!!!

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 

    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    X_norm=(X_norm-mu)/sigma
    # =========================== YOUR CODE HERE =====================

    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    Returns
    -------
    J : float
        The value of the cost function. 

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0]   # number of training examples

    # You need to return the following variable correctly
    J = 0
    Z=np.dot(X,theta.T)
    for i in range(m):
        J+=(Z[i]-y[i])**2
    J=J/(2*m)
    # ======================= YOUR CODE HERE ===========================

    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    alpha : float
        The learning rate for gradient descent. 

    num_iters : int
        The number of iterations to run gradient descent. 

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]   # number of training examples

    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        Z=np.dot(X,theta.T)
        delta=(np.dot((Z-y).T,X))*alpha/m
        theta=theta-delta
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history

def plotData(X, y,theta):
    """
    
    Plots the data points x and y into a new figure. Plots the data 
    points and gives the figure axes labels of population and profit.
    
    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.
    
    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.    
    
    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You 
    can also set the marker edge color using the `mec` property.
    """
    # surface plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:,1],X[:,2],y,marker='o',s=50,c='r',cmap='viridis')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('House price')
    x_axis=np.arange(min(X[:,1]),max(X[:,1]),1)
    y_axis=np.arange(min(X[:,1]),max(X[:,1]),1)
    ax.plot(x_axis,y_axis,(theta[0]+x_axis*theta[1]+y_axis*theta[2]),label='Linear regression')
    ax.legend()
    #fig = plt.figure()  # open a new figure
    plt.show()
    
def normalEqn(X, y):
    """
    Computes the closed-form solution to linear regression using the normal equations.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        The value at each data point. A vector of shape (m, ).

    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).

    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.

    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    theta = np.zeros(X.shape[1])
    theta=np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)
    # ===================== YOUR CODE HERE ============================
    return theta




# call featureNormalize on the loaded datay
X_norm, mu, sigma = featureNormalize(X)
# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

"""
Instructions
------------
We have provided you with the following starter code that runs
gradient descent with a particular learning rate (alpha). 

Your task is to first make sure that your functions - `computeCost`
and `gradientDescent` already work with  this starter code and
support multiple variables.

After that, try running gradient descent with different values of
alpha and see which one gives you the best result.

Finally, you should complete the code at the end to predict the price
of a 1650 sq-ft, 3 br house.

Hint
----
At prediction, make sure you do the same feature normalization.
"""
# Choose some alpha value - change this
theta = np.zeros(3)
alpha = 0.001
num_iters = 10000
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
print(theta)
theta=normalEqn(X,y)
print(theta)
# init theta and run gradient descent
'''
for i in range(5):
    alpha = 10**(-i)
    num_iters = int(100**(i/2))
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
    # Plot the convergence graph
    plt.plot(np.arange(len(J_history)), J_history,label=alpha,)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.legend()
    # Display the gradient descent's result
    print('theta computed from gradient descent: {:s}'.format(str(theta)))
    #plotData(X,y,theta)
    #print(theta)
'''

plt.show()
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)









grader[7] = normalEqn
grader.grade()