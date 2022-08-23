import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import argparse

class GradientBooster():

    def __init__(self, weak_learner, n_estimators=2):
        """ 

        Args:
          weak_learner: weak learner object. Must support fit(self, x, y), and predict(self, X) like sklearn classifiers
          n_estiamtors: int, number of estimators to construct
        """
        self.models = []
        self.alphas = []
        self.weak_learner = weak_learner
        self.n_estimators = n_estimators
    
    def fit(self, X, y, X_val, y_val, lr=0.1):
        """ 
        Gradient Boosting with least squares loss.

        Args:
        X: numpy array shape (n,d) - the data (rows)
        y: numpy array shape (n,) all elements in {-1, 1} - the labels
        X_val: numpy array shape (n,d) - the validation data (rows)
        y_val: numpy array shape (n,) all elements in {-1, 1} - the validation labels
        lr: learning rate
            
        Computes and stores 
          - models: lists of size n_estimators of week_learner
          - alphas: lists of size n_estimators of float

        Returns:
          train_scores: list of scores (accuracy) on the training data for each iteration of the algorithm (for every model considered)
          val_scores: list of scores (accuracy) on the validation data for each iteration of the algorithm (for every model considered)

        to create a weak learners use 
        tmp = self.weak_learner()
        tmp.fit(Data, values)
        """  
        train_scores = []
        val_scores = []
        ### YOUR CODE HERE
        wl = self.weak_learner()
        wl.fit(X, y)
        self.models.append(wl)
        self.alphas.append(1)
        for i in range(self.n_estimators-1):
            cur_train_score = self.score(X, y)
            train_scores.append(cur_train_score)
            cur_val_score = self.score(X_val, y_val)
            val_scores.append(cur_val_score)
            old_pred = self.predict(X)
            res = (y - old_pred)
            wl = self.weak_learner()
            wl.fit(X, res)
            #print(alphat)
            self.models.append(wl)
            self.alphas.append(lr)
            
        train_scores.append(self.score(X, y))
        cur_val_score = self.score(X_val, y_val)
        val_scores.append(cur_val_score)
        ### END CODE

        # remember to ensure that self.models and self.alphas are filled
        assert len(self.models) == self.n_estimators
        assert len(self.alphas) == self.n_estimators
        return train_scores, val_scores
        

        

    def predict(self, X):
        """ Compute the output prediction of the ensemble on the data (sum_i a_i h_i(x_i))

        Args:
        X: numpy array shape (n, d) - the data (rows)
        
        Returns:
          pred: np.array (n, ) ensemble output prediction on each input point in X (rows)
        """
        pred = None
        if len(self.models) == 0:
            return np.zeros(X.shape[0])
        ### YOUR CODE HERE 3-8 lines
        all_model_preds = [alpha * model.predict(X) for (alpha, model) in zip(self.alphas, self.models)]
        all_model_preds = np.c_[all_model_preds]
        pred = np.sum(all_model_preds, axis=0)
        ### END CODE
        return pred
        

    def score(self, X, y):
        """ Return accuracy of model on data X with labels y ((1/n) (sum_i (f(x_i) == y_i])^2)
        
        Args:
          X (numpy array shape n, d)
        returns
          score (float) classifier mean least squares loss on data X with labels y
        """
        score = 0
        ### YOUR CODE HERE 1-3 lines
        pred = self.predict(X)
        score = ((pred-y)**2).mean()
        ### END CODE
        return score


def test_housing(max_depth, random_state=42):
    from  sklearn.datasets import fetch_california_housing
    rdata = fetch_california_housing()
    Xr = rdata.data
    yr = rdata.target
    yr = yr - yr.mean()
    X_train, X_test, y_train, y_test = train_test_split(Xr, yr, test_size=0.40, random_state=random_state)
    print('First, lets test a standard regression tree')
    for d in [1, 7, 20]:
        tree = DecisionTreeRegressor(max_depth=d)
        tree.fit(X_train, y_train)
        print('Normal Regression Tree max depth: {0}'.format(d))
        print('Normal Regression Tree - Trainining data Least Squares Loss', ((tree.predict(X_train) - y_train)**2).mean())
        print('Normal Regression Tree Normal - Test data Least Squares Loss', ((tree.predict(X_test) - y_test)**2).mean())
    print('Lets see if we can do better\n')
    
    my_learner = lambda: DecisionTreeRegressor(max_depth=max_depth)
    bdt = GradientBooster(my_learner,
                            n_estimators=100)
    train_scores, val_scores = bdt.fit(X_train, y_train, X_test, y_test)
    print('Training Data, Final Least Squares Loss:', train_scores[-1])
    print('Validation Test Data, Least Squares Loss final model:', val_scores[-1])
    print('Validation Test Data, Least Squares Loss best_model:', np.min(val_scores))

    fig, ax = plt.subplots(1, 1, figsize=(12,10))
    ax.plot(train_scores, 'b--', label='L2 loss train', linewidth=2)
    ax.plot(val_scores, 'r--', label='L2 loss validation', linewidth=2)
    ax.legend(fontsize=15)
    ax.set_title('Loss Per Iteration for Gradient Boosting - Max Depth {0}'.format(max_depth), fontsize=20)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-max_depth', default=1, type=int)
    parser.add_argument('-random_state', default=42, type=int)

    args = parser.parse_args()
    print('Testing Gradient Boosting with Regression Trees with Max Depth {0}'.format(args.max_depth))
    test_housing(args.max_depth, args.random_state)
    
