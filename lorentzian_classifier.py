"""
Lorentzian Distance-based k-Nearest Neighbors Classifier.
More robust to outliers than Euclidean distance.
"""
import numpy as np

class LorentzianClassifier:
    """
    k-Nearest Neighbors classifier using Lorentzian distance metric.
    
    Lorentzian distance: D(x, y) =  log(1 + |x_i - y_i|)
    More robust to outliers than Euclidean distance.
    """
    
    def __init__(self, k=8, weight_by_distance=True):
        """
        Args:
            k: Number of nearest neighbors to consider
            weight_by_distance: If True, closer neighbors have more influence
        """
        self.k = k
        self.weight_by_distance = weight_by_distance
        self.X_train = None
        self.y_train = None
        
    def lorentzian_distance(self, x1, x2):
        """
        Calculate Lorentzian distance between two points.
        More robust to outliers than Euclidean distance.
        """
        return np.sum(np.log(1 + np.abs(x1 - x2)))
    
    def fit(self, X, y):
        """Store training data (lazy learning - no actual training)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict_proba_single(self, x):
        """Predict probability for a single sample."""
        # Calculate distances to all training samples
        distances = np.array([
            self.lorentzian_distance(x, x_train) 
            for x_train in self.X_train
        ])
        
        # Find k nearest neighbors
        k_indices = np.argpartition(distances, min(self.k, len(distances)-1))[:self.k]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]
        
        # Weight by distance if enabled
        if self.weight_by_distance:
            # Avoid division by zero
            weights = 1.0 / (k_distances + 1e-6)
            weights = weights / weights.sum()
            
            # Weighted average of neighbor labels
            prob_positive = np.sum(weights * k_labels)
        else:
            # Simple average
            prob_positive = np.mean(k_labels)
        
        return np.array([1 - prob_positive, prob_positive])
    
    def predict_proba(self, X):
        """Predict probabilities for all samples."""
        probas = np.array([self.predict_proba_single(x) for x in X])
        return probas
    
    def predict(self, X):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

