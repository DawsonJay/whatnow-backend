#!/usr/bin/env python3
"""
Lightweight SGD Classifier implementation using only numpy and scipy.
Replaces scikit-learn's SGDClassifier for deployment compatibility.
"""

import numpy as np
from typing import List, Optional, Tuple


class LightweightSGDClassifier:
    """
    Lightweight SGD Classifier implementation using only numpy and scipy.
    Provides the same interface as sklearn's SGDClassifier for contextual bandits.
    """
    
    def __init__(self, learning_rate='adaptive', eta0=0.01, random_state=42, loss='log_loss'):
        """
        Initialize the lightweight SGD classifier.
        
        Args:
            learning_rate: Learning rate schedule ('adaptive' or 'constant')
            eta0: Initial learning rate
            random_state: Random seed for reproducibility
            loss: Loss function ('log_loss' for logistic regression)
        """
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.random_state = random_state
        self.loss = loss
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.is_fitted = False
        
        # For adaptive learning rate
        self.learning_rate_ = eta0
        self.t_ = 0  # Time step counter
        
        # Set random seed
        np.random.seed(random_state)
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _log_loss_gradient(self, X, y, coef, intercept):
        """Compute gradient for logistic loss."""
        # Compute predictions
        z = X @ coef + intercept
        p = self._sigmoid(z)
        
        # Compute gradient
        error = p - y
        coef_grad = X.T @ error / len(X)
        intercept_grad = np.mean(error)
        
        return coef_grad, intercept_grad
    
    def partial_fit(self, X, y, classes=None):
        """
        Partial fit for online learning.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            classes: Class labels (for first fit)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Initialize model if first fit
        if not self.is_fitted:
            if classes is None:
                classes = np.unique(y)
            self.classes_ = np.array(classes)
            
            n_features = X.shape[1]
            n_classes = len(classes)
            
            # Initialize weights
            self.coef_ = np.random.normal(0, 0.01, (n_classes, n_features))
            self.intercept_ = np.zeros(n_classes)
            
            self.is_fitted = True
        
        # Convert y to binary if needed
        if len(self.classes_) == 2:
            # Binary classification
            y_binary = (y == self.classes_[1]).astype(int)
            
            # Update learning rate if adaptive
            if self.learning_rate == 'adaptive':
                self.learning_rate_ = self.eta0 / (1 + self.t_)
            
            # Compute gradient
            coef_grad, intercept_grad = self._log_loss_gradient(X, y_binary, self.coef_[1], self.intercept_[1])
            
            # Update weights
            self.coef_[1] -= self.learning_rate_ * coef_grad
            self.intercept_[1] -= self.learning_rate_ * intercept_grad
            
            # Update time step
            self.t_ += 1
    
    def decision_function(self, X):
        """
        Compute decision function.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Decision function values
        """
        if not self.is_fitted:
            return np.zeros(X.shape[0])
        
        X = np.array(X)
        
        if len(self.classes_) == 2:
            # Binary classification
            return X @ self.coef_[1] + self.intercept_[1]
        else:
            # Multi-class (not implemented for simplicity)
            return X @ self.coef_[0] + self.intercept_[0]
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            return np.ones((X.shape[0], len(self.classes_))) / len(self.classes_)
        
        X = np.array(X)
        decision = self.decision_function(X)
        
        if len(self.classes_) == 2:
            # Binary classification
            prob_positive = self._sigmoid(decision)
            return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Multi-class (not implemented for simplicity)
            return np.ones((X.shape[0], len(self.classes_))) / len(self.classes_)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            return np.full(X.shape[0], self.classes_[0])
        
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_params(self, deep=True):
        """Get model parameters."""
        return {
            'learning_rate': self.learning_rate,
            'eta0': self.eta0,
            'random_state': self.random_state,
            'loss': self.loss
        }
    
    def set_params(self, **params):
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
