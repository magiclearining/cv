from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    score = X.dot(W)
    score_max = np.max(score, axis=1,keepdims = True)
    exp_score = np.exp(score - score_max)
    sum_exp_score = np.sum(exp_score, axis=1, keepdims=True)
    prob = exp_score / sum_exp_score

    for i in xrange(num_train):
      for j in xrange(num_class):
        if (y[i] == j):
          loss += -np.log(prob[i,j])
          dW[:,j] -= X[i] * (1 - prob[i,j])
        else:
          dW[:,j] += X[i] * prob[i,j]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW = dW / num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # dev data shape:  (500, 3073)
    # dev labels shape:  (500,)
    # W = np.random.randn(3073, 10) * 0.0001
    num_train = X.shape[0]
    num_class = W.shape[1]

    score = X.dot(W)
    score_max = np.max(score,axis=1,keepdims=True)
    exp_score = np.exp(score-score_max)
    sum_exp_score = np.sum(exp_score,axis=1,keepdims=True)
    prob = exp_score / sum_exp_score

    keepProb = np.zeros_like(prob)
    keepProb[np.arange(num_train), y] = 1.0
    loss = np.sum(-np.log(prob[np.arange(0,num_train),y])) / num_train + 0.5 * reg * np.sum(W*W)
    dW += -np.dot(X.T, keepProb - prob) / num_train + reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
