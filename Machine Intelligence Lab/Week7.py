import numpy as np
from sklearn.tree import DecisionTreeClassifier

"""
Use DecisionTreeClassifier to represent a stump.
------------------------------------------------
DecisionTreeClassifier Params:
    critereon -> entropy
    max_depth -> 1
    max_leaf_nodes -> 2
Use the same parameters
"""
# REFER THE INSTRUCTION PDF FOR THE FORMULA TO BE USED 

class AdaBoost:

    """
    AdaBoost Model Class
    Args:
        n_stumps: Number of stumps (int.)
    """

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        """
        Fitting the adaboost model
        Args:
            X: M x D Matrix(M data points with D attributes each)(numpy float)
            y: M Vector(Class target for all the data points as int.)
        Returns:
            the object itself
        """
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)

        return self

    def sumFunc213 (self, sample_weights):
        n = 0
        for i in sample_weights:
                if isinstance(i, list):
                        n += self.sumFunc213(i)
                else:
                        n += i
        return n
    
    def stump_error(self, y, y_pred, sample_weights):
        """
        Calculating the stump error
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
        Returns:
            The error in the stump(float.)
        """

        # TODO
        # pass
        
        return (self.sumFunc213(sample_weights*np.where(y_pred!=y,1,0)))/(self.sumFunc213(sample_weights))

        # DONE

    def compute_alpha(self, error):
        """
        Computing alpha
        The weight the stump has in the final prediction
        Use eps = 1e-9 for numerical stabilty.
        Args:
            error:The stump error(float.)
        Returns:
            The alpha value(float.)
        """
        eps = 1e-9
        # TODO
        # pass

        return (np.log((1-error)/(eps+error)))/2

        # DONE

    def updateHelper213 (self, y, y_pred, sample_weights, e_213):
        updated_213 = []

        for i in range(len(sample_weights)):
                if y[i] == y_pred[i]:
                        updated_213.append(sample_weights[i]/(2*e_213))
                else:
                        updated_213.append(sample_weights[i]/(2*(1-e_213)))

        return updated_213
    
    def neqFunc213 (self, y, y_pred):
        for i in range(len(y)):
                if isinstance(y[i], list):
                        return self.neqFunc213(y[i], y_pred[i])
                else:
                        return (y[i] != y_pred[i])
    
    def update_weights(self, y, y_pred, sample_weights, alpha):
        """
        Updating Weights of the samples based on error of current stump
        The weight returned is normalized
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
            alpha: The stump weight(float.)
        Returns:
            new_sample_weights:  M Vector(new Weight of each sample float.)
        """

        # TODO
        # pass

        e_213 = 1 - self.stump_error(y, y_pred, sample_weights)
        if(e_213 == 1):
                return (sample_weights)*np.exp(alpha*(self.neqFunc213(y, y_pred)).astype(int))

        return self.updateHelper213(y, y_pred, sample_weights, e_213)
        # DONE

    def predHelper213 (self, X):
        for stump in range(self.n_stumps):
                return self.stumps[stump].predict(X)
    
    def predict(self, X):
        """
        Predicting using AdaBoost model with all the decision stumps.
        Decison stump predictions are weighted.
        Args:
            X: N x D Matrix(N data points with D attributes each)(numpy float)
        Returns:
            pred: N Vector(Class target predicted for all the inputs as int.)
        """
        # TODO
        # pass
        return np.sign((np.array([self.predHelper213(X)]))[0])

        # DONE

    def evaluate(self, X, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix
            y: True target of test data
        Returns:
            accuracy : (float.)
        """
        pred = self.predict(X)
        # find correct predictions
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy
