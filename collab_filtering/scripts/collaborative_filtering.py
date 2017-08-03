import pandas as pd
import numpy as np
from numpy.random import randn
from scipy.optimize import minimize

class Collaborative_Filtering(object):
    def __init__(self, nparams, train_data, cost, pred_file=None):
        self.nparams = nparams
        self.pred_file = pred_file
        self.train_data = train_data
        self.my_ratings = None
        self.cost = cost
        self.Y_train = None
        self.Y_mean = None
        self.inital_parameters = None
        self.n_artists = None
        self.n_users = None
        self.X = None
        self.Theta = None
        self.J_cost_history = None
        self.prediction = None 
        self.final_cost = None
   
    def add_users(self):
        if self.pred_file is not None:
            self.my_ratings = pd.read_csv(self.pred_file, sep=";", index_col=0)
            new_users = self.my_ratings.columns[1:]

            # Appending columns at the end of train_data
            # with the values we want to rate
            for user in new_users:
               self.train_data[user] = np.repeat(np.nan, self.n_artists)

            # Set each new rated artist in the train_data
            for index in self.my_ratings.index:
                for user in new_users:
                    self.train_data[user].loc[index] = self.my_ratings[user].loc[index]
                
            # Updating number of artists and number of users
            self.n_artists, self.n_users = self.train_data.shape

    def initalize_model(self):
        self.n_artists, self.n_users = self.train_data.shape
        self.add_users()
        self.X = randn(self.n_artists, self.nparams)
        self.Theta = randn(self.n_users, self.nparams)
        self.Y_train, self.Y_mean = self.rating_normalize(self.train_data)
        self.inital_parameters = np.concatenate((self.X.flatten(), self.Theta.flatten()))

    def rating_normalize(self, X):
        """Given a training matrix X, divide compute the mean of each
        row and return a list with the mean and X less the mean"""
        X_mean = np.mean(np.nan_to_num(X), axis=1, keepdims=True)
        X_train = X.values - X_mean

        return X_train, X_mean


    def cost_function(self, error, X, Theta):
        J = np.sum(np.nansum(error ** 2)) / 2
        J += self.cost / 2 * (sum(sum(Theta ** 2)) + sum(sum(X ** 2)))

        return J


    def compute_gradients(self, error, Theta, X):
        # replacing np.nan with 0
        grads = np.nan_to_num(error)
        X_grad = np.dot(grads, Theta) + self.cost * X
        Theta_grad = np.dot(grads.T, X) + self.cost * Theta

        return X_grad, Theta_grad


    def collaborative_filtering(self, parameters):
        border = self.n_artists * self.nparams

        # Unrolling the parameters
        X = np.reshape(parameters[:border], (self.n_artists, self.nparams))
        Theta = np.reshape(parameters[border:], (self.n_users, self.nparams))
        error = np.dot(X, Theta.T) - self.Y_train

        # Cost Function
        J = self.cost_function(error, X, Theta)
        # Gradient computation
        X_grad, Theta_grad = self.compute_gradients(error, Theta, X)

        # Rolling Parameters
        gradients = np.concatenate((X_grad.flatten(), Theta_grad.flatten()))

        return (J, gradients)

    def train(self, record_J=False, display=False, max_iter=200):
        if record_J:
            self.J_cost_history = []
            def J_hist(xk): self.J_cost_history.append(self.collaborative_filtering(xk)[0])
        else:
            J_hist = None

        estimate = minimize(self.collaborative_filtering, self.inital_parameters, method="L-BFGS-B", jac=True,
                            options={"disp":display, "maxiter":max_iter}, callback=J_hist) 

        border = self.n_artists * self.nparams
        self.X = np.reshape(estimate.x[:border], (self.n_artists, self.nparams))
        self.Theta = np.reshape(estimate.x[border:], (self.n_users, self.nparams))
        self.final_cost = estimate

        self.prediction = np.dot(self.X, self.Theta.T) + self.Y_mean
