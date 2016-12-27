from __future__ import print_function
import numpy as np


class TwoLayerNet(object):

    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #
        # TODO: Perform the forward pass, computing the class scores for the input.
        # Store the result in the scores variable, which should be an array of
        # shape (N, C).
        #

        # first layer (fully connected)
        scores1 = X.dot(W1) + b1

        # ReLU
        activations1 = np.copy(scores1)
        activations1[scores1 < 0] = 0

        # second layer (fully connected)
        scores = activations1.dot(W2) + b2

        #
        # END OF YOUR CODE
        #

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #
        # TODO: Finish the forward pass, and compute the loss. This should include
        # both the data loss and L2 regularization for W1 and W2. Store the result
        # in the variable loss, which should be a scalar. Use the Softmax
        # classifier loss. So that your results match ours, multiply the
        # regularization loss by 0.5
        #

        exp_scores = np.exp(scores)
        norm = np.sum(exp_scores, axis=1)
        log_prob = exp_scores[range(N), y] / norm
        loss = np.mean(-np.log(log_prob))

        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        #
        # END OF YOUR CODE
        #

        # Backward pass: compute gradients
        #
        # TODO: Compute the backward pass, computing the derivatives of the weights
        # and biases. Store the results in the grads dictionary. For example,
        # grads['W1'] should store the gradient on W1, and be a matrix of same size
        #

        H = W1.shape[1]
        C = scores.shape[1]

        assert X.shape == (N, D)
        assert y.shape == (N,)
        assert scores1.shape == (N, H)
        assert activations1.shape == (N, H)
        assert scores.shape == (N, C)
        assert norm.shape == (N,)
        assert W1.shape == (D, H)
        assert b1.shape == (H,)
        assert W2.shape == (H, C)
        assert b2.shape == (C,)

        dW1 = np.zeros_like(W1)
        dW2 = np.zeros_like(W2)
        db1 = np.zeros_like(b1)
        db2 = np.zeros_like(b2)

        # Backprop into scores
        flags = np.equal(y[:, np.newaxis],
                         np.arange(C)[np.newaxis, :]).astype(np.float)
        dscores = -flags + exp_scores / norm[:, np.newaxis]

        assert dscores.shape == (N, C)

        # Backprop into W2. The local gradient with respect to W2 is just
        # activations1, so we have to multiply by that.
        dW2 = np.dot(activations1.T, dscores) / N
        dW2 += reg * W2

        # Backprop into b2. The local gradient is just 1, so this is equal
        # to dscores, averaged over all training samples.
        db2 = np.mean(dscores, axis=0)

        # Backprop into activations1 (local gradient is W2)
        dactivations1 = np.dot(dscores, W2.T)

        assert dactivations1.shape == (N, H)

        # Backprop into scores1 (local gradient is one, except where
        # scores1 < 0.
        dscores1 = np.copy(dactivations1)
        dscores1[scores1 < 0] = 0

        # Backprop into W1
        dW1 = np.dot(X.T, dscores1) / N
        dW1 += reg * W1

        # Backprop into b1
        db1 = np.mean(dscores1, axis=0)

        grads = {
            "W1": dW1,
            "W2": dW2,
            "b1": db1,
            "b2": db2
        }

        #
        # END OF YOUR CODE
        #

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #
            # TODO: Create a random minibatch of training data and labels, storing
            # them in X_batch and y_batch respectively.
            #
            batch_ind = np.random.choice(num_train, size=batch_size, replace=True)
            X_batch = X[batch_ind, :]
            y_batch = y[batch_ind]
            #
            # END OF YOUR CODE
            #

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #
            # TODO: Use the gradients in the grads dictionary to update the
            # parameters of the network (stored in the dictionary self.params)
            # using stochastic gradient descent. You'll need to use the gradients
            # stored in the grads dictionary defined above.
            #

            for key in self.params.keys():
                self.params[key] -= learning_rate * grads[key]

            #
            # END OF YOUR CODE
            #

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning
            # rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = np.equal(self.predict(X_batch), y_batch).astype(np.float).mean()
                val_acc = np.equal(self.predict(X_val), y_val).astype(np.float).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        #
        # TODO: Implement this function; it should be VERY simple!
        #

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # first layer (fully connected)
        scores1 = X.dot(W1) + b1

        # ReLU
        activations1 = np.copy(scores1)
        activations1[scores1 < 0] = 0

        # second layer (fully connected)
        scores = activations1.dot(W2) + b2

        y_pred = np.argmax(scores, axis=1)

        #
        # END OF YOUR CODE
        #

        return y_pred