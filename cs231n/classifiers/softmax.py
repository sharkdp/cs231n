from __future__ import print_function
import numpy as np


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

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful  #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                                        #
    ##########################################################################

    num_train = X.shape[0]

    for i in xrange(num_train):
        scores = np.dot(X[i, :], W)

        # for numerical stability, does not change the result:
        scores -= np.max(scores)

        exp_scores = np.exp(scores)
        norm = np.sum(exp_scores)
        loss += -np.log(exp_scores[y[i]] / norm)

        # gradient
        dW[:, y[i]] -= X[i, :]
        dW += np.outer(X[i, :], exp_scores / norm)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * W

    #######################################################################
    #                          END OF YOUR CODE                           #
    #######################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    # Inputs:
    # - W: A numpy array of shape (D, C) containing weights.
    # - X: A numpy array of shape (N, D) containing a minibatch of data.
    # - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    #   that X[i] has label c, where 0 <= c < C.
    # - reg: (float) regularization strength

    # scores ~ (N, C)
    # norm ~ (N, )
    # exp_scores[y].T / norm ~ (C, N)

    num_train = X.shape[0]

    scores = np.dot(X, W)

    # for numerical stability, does not change the result:
    # scores -= np.max(scores, axis=0)

    exp_scores = np.exp(scores)
    norm = np.sum(exp_scores, axis=1)
    log_prob = exp_scores[range(num_train), y] / norm
    loss = np.sum(-np.log(log_prob))

    # print("W ~ ", W.shape)
    # print("X ~ ", X.shape)
    # print("exp_scores ~ ", exp_scores.shape)
    # print("norm ~ ", norm.shape)
    # print("y ~ ", y.shape)
    # print("dW[:, y] ~ ", dW[:, y].shape)
    # print("X[range(num_train), :] ~", X[range(num_train), :].shape)

    # gradient

    # TODO: how to vectorize this loop?
    for i in range(num_train):
        dW[:, y[i]] -= X[i, :]

    # dW[:, y] -= X.T  # this does not work :-(

    dW += np.dot(exp_scores.T / norm, X).T

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * W
    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW
