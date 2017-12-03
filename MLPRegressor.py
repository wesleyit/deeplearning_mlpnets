""" MLPRegressor - A regression multi layer perceptron
Author: Wesley Silva
Github: wesleyit
2017
"""


# All needed libraries are imported here.
import random as rnd
import numpy as np


class MLPRegressor:
    """ MLPRegressor - A regression multi layer perceptron
    -----------------------------------------------------------

    Hello! This is a simple implementation of a multi layered perceptron
    for regression tasks. It is bundled with 1 input layer, 1 hidden
    layer and 1 output layer. You can choose the neurons in each layer
    and set the hyperparameters.

    It is a regression neural net because the output layer uses a
    linear output able to return any numerical value.
    This behavior is different from the regression neural network,
    which features a sigmoid function, which can return an output
    from 0.0 to 1.0. This value may be rouded, as well.
    """


    def sigmoid(self, x):
        """ A sigmoid function which squeezes a value between 0 and 1.

        Args:
            x: The input number. Must be a number or an array of numbers.

        Returns:
            A float between 0.0 and 1.0, or an array of floats in this range.
        """
        return .5 * (1 + np.tanh(.5 * x))


    def debug_msg(self, msg, lvl):
        """ A function to print debug messages on the console.

        Args:
            msg: A string to be printed on the console.
            lvl: The level to show messages.

        Returns:
            Nothing.
        """
        if self.debug >= lvl:
            print(">>> %s" % msg)


    def __init__(self, n_inputs, n_hidden, n_outputs, rate, epochs,
                 w_std_d=0.5, round=True, debug=0):
        """ Initializes the neural network hyperparameters.

        Args:
            n_inputs (int): How many neurons on the input layer.
            n_hidden (int): How many neurons on the hidden layer.
            n_outputs (int): How many neurons on the output layer.
            rate (float): The learning rate.
            epochs (int): The number of training iterations.
            w_std_d (float): The standard deviation from 0 to randomize the weights.
            round (bool): If the output must be round.
            debug (int): The more the level, the more messages are displayed.

        Returns:
            A MPPRegressor object ready to be trained.
        """
        self.w1 = np.array([rnd.uniform(-w_std_d, w_std_d) for _ in range(n_inputs * n_hidden)])
        self.w1 = self.w1.reshape(n_inputs, n_hidden)
        self.w2 = np.array([rnd.uniform(-w_std_d, w_std_d) for _ in range(n_hidden * n_outputs)])
        self.w2 = self.w2.reshape(n_hidden, n_outputs)
        self.rate = rate
        self.epochs = epochs
        self.round = round
        self.debug = debug
        self.debug_msg("New neural network created with Hyperparameters:", 2)
        self.debug_msg("Weights \nw1:%s\nw2:%s" % (self.w1, self.w2), 2)
        self.debug_msg("Learning rate: %s, Epochs: %s" % (self.rate, self.epochs), 2)


    def feed_forward_step(self, X):
        """ Pass the input values through the weights and activations.

        Args:
            X: The input matrix. The shape must be [X, Y], where X is the rows and
            Y is the number of features.

        Returns:
            An array [X, Y], where X is the number of rows and Y are the ouputs.
        """
        results = []
        feed_forward = X
        feed_forward = np.dot(feed_forward, self.w1)
        feed_forward = self.sigmoid(feed_forward)
        self.debug_msg("Hidden layer result: %s" % feed_forward, 3)
        results.append(feed_forward)
        feed_forward = np.dot(feed_forward, self.w2)
        self.debug_msg("Output layer result: %s" % feed_forward, 3)
        results.append(feed_forward)
        return results


    def fit(self, X, y):
        """ Trains the network using backpropagation with Gradient Descent

        Args:
            X: The input matrix. The shape must be [R, F], where R is the rows and
            F is the number of features.
            y: The labels (the right answers). The shape must be [R, F].

        Returns:
            Nothing.
        """
        n_records, n_features = X.shape
        delta_w2 = 0
        delta_w1 = 0
        self.debug_msg("Fiting the model with %s rows and %s features" % (n_records, n_features), 2)
        for epoch in range(self.epochs):
            self.debug_msg("-" * 60, 4)
            self.debug_msg("Training process - iteration %s" % epoch, 4)

            # Error Gathering
            (yhat_hidden, yhat) = self.feed_forward_step(X)
            error = y - yhat
            self.mse = np.mean(((error) ** 2))
            if epoch % (self.epochs / 10) == 0:
                self.debug_msg("mse: %s" % round(self.mse, 5), 1)
            self.debug_msg("yhat - hidden: %s, output: %s" % (list(yhat_hidden), list(yhat)), 4)

            # There is no need to derivate the activation function since
            # it is a linear output.
            grad_w2 = error

            # The delta_w2 must have the shape [H, O], where H is the number
            # of neurons on the hidden layer and O is the number of neurons
            # on the output layer.
            delta_w2 += np.dot(yhat_hidden.T, grad_w2)

            # Gets the hidden layer prime. The shape is [N, H], where
            # N in the number of rows and H is the number of neurons on
            # the hidden layer.
            yhat_hidden_prime = yhat_hidden * (1 - yhat_hidden)

            # Gets the error of the first set of weights. The error must
            # have the shape [N, H], where N in the number of rows and
            # H is the number of neurons on the hidden layer.
            w1_error = np.dot(grad_w2, self.w2.T)

            # Here is simple mult, not dot product. The shape will remain the same,
            # [N, H].
            grad_w1 = w1_error * yhat_hidden_prime

            # The delta_w1 will have the shape [I, H], where I is the number
            # of neurons on the input layer and H is the number of neurons
            # on the hidden layer.
            delta_w1 = delta_w1 + np.dot(X.T, grad_w1)

            # Weights updates, a simple multiplication, not the dot product.
            self.w2 = self.w2 + self.rate * delta_w2 / n_records
            self.w1 = self.w1 + self.rate * delta_w1 / n_records
        self.debug_msg("Fit completed!", 2)
        self.debug_msg("-" * 60, 4)


    def predict(self, X):
        """ Predicts values using a trained network.

        Args:
            X: A [X, Y] matrix with X rows of Y features.

        Returns:
            A [X, Y] matrix with X rows of Y outputs.
        """
        _, yhat = self.feed_forward_step(X)
        if self.round:
            return yhat.round(5)
        else:
            return yhat
