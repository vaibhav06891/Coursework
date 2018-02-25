import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_class = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in xrange(num_train):
        weighted_sum = X[i].dot(W)
        weighted_sum -= np.max(weighted_sum)
        scores = np.exp(weighted_sum)
        total = sum(scores)
        loss -= np.log(scores[y[i]]/total)
#         X_T = np.reshape(X[i],(X.shape[1], 1))
#         print X_T.shape, dW[:, 3].shape
#         for j in range(num_class):
#             if j == y[i]:
#                 dW[:, j] += X[i]*(scores[j]/total - 1)
#             else:
#                 dW[:, j] += X[i] * scores[j]/total
        dW += np.reshape(X[i],(X.shape[1], 1))*scores/total
        dW[:, y[i]] -= X[i]
        
    loss /= num_train
    dW /= num_train
    loss += 0.5*reg*np.sum(np.square(W))
    dW += reg*W
#     print sum(W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    weighted_sum = X.dot(W)
    weighted_sum -= np.max(weighted_sum)
    scores = np.exp(weighted_sum)
    total = np.sum(scores, axis = 1)
    prob = scores/np.reshape(total, (total.shape[0],1))
    dW = X.T.dot(prob)
    mask = np.zeros_like(scores)
    mask[np.arange(num_train), y] = 1
    
    dW -= X.T.dot(mask)
    loss = np.sum(-np.log(prob[np.arange(num_train), y]))
    
    
    loss /= num_train
    dW /= num_train
    loss += 0.5*reg*np.sum(np.square(W))
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

