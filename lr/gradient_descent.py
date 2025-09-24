from typing import List, Optional

import numpy as np
from .base import Model

class LogisticRegressionGD(Model):
    """
    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for gradient descent updates
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent
    tol : float, default=1e-6
        Tolerance for convergence criterion
    batch_size : int, default=32
        Size of mini-batches for gradient descent
    random_state : int, optional
        Random seed for reproducibility
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 batch_size: int = 32,
                 random_state: Optional[int] = None,
                 fit_intercept: bool = True):
        super().__init__(fit_intercept=fit_intercept)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.cost_history_: List[float] = []
        self.n_iter_: Optional[int] = None

    def cost_function(self, X: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray] = None) -> float:
        """
        Compute the logistic regression cost function (log-loss).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with intercept column if fit_intercept=True
        y : np.ndarray
            Target values
        theta : np.ndarray, optional
            Parameter vector. If None, uses current model parameters.

        Returns
        -------
        cost : float
            Log-loss cost value
        """
        if theta is None:
            if self.coef_ is None:
                raise ValueError("Model not fitted yet")
            # Reconstruct theta from fitted parameters
            if self.fit_intercept:
                theta = np.vstack([[[self.intercept_]], self.coef_])
            else:
                theta = self.coef_
        
        m = X.shape[0]  
        predictions = X @ theta
        predictions = self.sigmoid(predictions)
        # Prevent log(0) by clipping
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1-y) * np.log(1-predictions))
        return cost

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the cost function.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with intercept column if fit_intercept=True
        y : np.ndarray
            Target values
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        gradients : np.ndarray
            Gradient vector
        """
        m = X.shape[0]
        predictions = X @ theta
        # added sigmoid to return probablity along 0 to 1
        predictions = self.sigmoid(predictions)
        error = predictions - y.reshape(-1, 1)
        gradients = (1/m) * X.T @ error
        return gradients

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionGD':
        """
        Fit the model using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : LogisticRegressionGD
            Returns self for method chaining
        """
        np.random.seed(self.random_state)

        # Validate and preprocess input
        X, y = self._validate_input(X, y)
        
        # Add intercept column if needed
        X = self._add_intercept(X)
        
        # Initialize theta to zeros (safe start)
        theta = np.zeros((X.shape[1], 1))
            
        for i in range(self.max_iter):
            # current cost
            current_cost = self.cost_function(X, y, theta)
            self.cost_history_.append(current_cost)

            # which way do I move
            gradients = self._compute_gradients(X, y, theta)
            
            # update thetas
            theta = theta - self.learning_rate * gradients
            
            # Check for NaN
            if np.isnan(theta).any():
                print(f"NaN detected at iteration {i}")
                break
            
            if len(self.cost_history_) > 1:
                cost_change = abs(self.cost_history_[-2] - current_cost)
                if cost_change < self.tol:
                    break
                
        if self.fit_intercept:
            self.intercept_ = theta[0, 0]  
            self.coef_ = theta[1:, :]     
        else:
            self.coef_ = theta
        
        self.n_iter_ = i + 1

        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities (sigmoid output)."""
        if self.coef_ is None:
            raise ValueError("model not fitted yet")
        
        X, _ = self._validate_input(X)
        X = self._add_intercept(X)
        
        if self.fit_intercept:
            linear_pred = X[:, 1:] @ self.coef_ + self.intercept_
        else:
            linear_pred = X @ self.coef_
        
        return self.sigmoid(linear_pred).flatten()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
