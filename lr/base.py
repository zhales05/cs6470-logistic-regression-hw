from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

class Model(ABC):
    """
    Abstract base class for machine learning models.
    
    This class defines the common interface for classification models
    as required by the rubric.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and preprocess input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features
        y : array-like of shape (n_samples,), optional
            Target values
            
        Returns
        -------
        X_processed : np.ndarray
            Processed feature matrix
        y_processed : np.ndarray or None
            Processed target vector
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        
        # Validate X shape
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        # Validate y shape if provided
        if y is not None:
            if y.ndim != 1:
                raise ValueError(f"y must be 1D array, got {y.ndim}D")
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        # Check for NaN/infinite values
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if y is not None and not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values")
        
        # Set number of features
        self.n_features_in_ = X.shape[1]
        
        return X, y
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to feature matrix if fit_intercept is True."""
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack([ones, X])
        return X
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        """
        Train the model on input features X and target y.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : Model
            Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input features X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        pass
    
    @abstractmethod
    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost (e.g., log-loss) for the current model parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        cost : float
            Cost value
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy score for classification.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True class labels
            
        Returns
        -------
        score : float
            Accuracy score (proportion of correct predictions)
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)


