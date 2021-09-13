import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pdb 
from sklearn.tree import DecisionTreeRegressor
from io import StringIO  
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydotplus


def plot_tree(dtree, feature_names):
    """ helper function """
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    print('exporting tree to dtree.png')
    graph.write_png('dtree.png')


class RegressionStump():
    
    def __init__(self):
        """ The state variables of a stump"""
        self.idx = None
        self.val = None
        self.left = None
        self.right = None
    
    def fit(self, data, targets):
        """ Fit a decision stump to data
        
        Find the best way to split the data in feat  minimizig the cost (0-1) loss of the tree after the split 
    
        Args:
           data: np.array (n, d)  features
           targets: np.array (n, ) targets
    
        sets self.idx, self.val, self.left, self.right
        """
        # update these three
        self.idx = 0
        self.val = None
        self.left = None
        self.right = None
        ### YOUR CODE HERE
        # i have added a slow and a fast version
        
        num_points, num_features = data.shape
        # print('num points, num_features', num_points, num_features)
        
        def feat_score(feat_idx):
            feat = data[:, feat_idx].copy()
            perm = np.argsort(feat)
            s_feat = feat[perm]
            s_targets = targets[perm]
            target_var = ((s_targets - s_targets.mean())**2).sum()
            s_left, s_right = sum_squares(s_targets)
            def score(idx, _vals):
                ##  slow version
                #left = _vals[0:idx]
                #right  = _vals[idx:]
                #assert len(left) + len(right) == len(_vals), (len(left), len(right), len(_vals))
                #left_mean = np.mean(left)
                #right_mean = np.mean(right)
                #left_error = np.sum((left-left_mean)**2)
                #assert np.allclose(left_error, s_left[idx])                                
                #right_error = np.sum((right-right_mean)**2)
                #assert np.allclose(right_error, s_right[idx])
                # return left_error+right_error
                # fast version
                return s_left[idx] + s_right[idx]
            # score for every split
            scores = np.array([score(x, s_targets) for x in range(0, num_points)])
            assert scores.min() <= target_var, target_var
            best_score_idx = np.argmin(scores)
            best_score = scores[best_score_idx]
            val = s_feat[best_score_idx]
            # print('best score', feat_idx, best_score, best_score_idx, val, s_feat[best_score_idx+1])
            
            return best_score, {'val': val, 
                                'left': np.mean(s_targets[:best_score_idx]), 
                                'right': np.mean(s_targets[best_score_idx:])
                                }       

        split_scores = []
        for f in range(0, num_features):
            total_score, _params = feat_score(f)
            split_scores.append(total_score)
            # print('score of {0} - {1}'.format(feat_names[f], total_score))
        # print('feature scores:', np.array(split_scores))
        best_feat = np.argmin(split_scores)
        best_score = split_scores[best_feat]
        # print('Best Feature idx: {0} - Best Cost: {1}'.format(best_feat, best_score))
        score_again, params = feat_score(best_feat)
        # print('double check score', score_again, best_score)
        self.idx = best_feat
        self.val = params['val']
        self.left = params['left']
        self.right = params['right']
        print("idx={}, val={}, left={}, right={}".format(self.idx, self.val, self.left, self.right))
        assert not np.isnan(self.left)
        assert not np.isnan(self.right)
        ### END CODE

    def predict(self, X):
        """ Regression tree prediction algorithm

        Args
            X: np.array, shape n,d
        
        returns pred: np.array shape n,  model prediction on X
        """
        pred = None
        ### YOUR CODE HERE
        dat = X[:, self.idx]
        n = X.shape[0]        
        decision = (dat < self.val).astype(int)
        # values = np.c_[np.ones(n)*self.left, np.ones(n)*self.right]
        pred = decision * self.left + (1-decision) * self.right
        ### END CODE
        return pred
    
    def score(self, X, y):
        """ Compute accuracy of model

        Args
            X: np.array, shape n,d
            y: np.array, shape n, 

        returns out: scalar - mean least squares scores cost over the data X, y 
        """
        out = None
        ### YOUR CODE HERE
        pred = self.predict(X)
        assert pred.shape == y.shape
        out = ((pred-y)**2).mean()
        ### END CODE
        return out
        

### YOUR CODE HERE
def sum_squares(X):
    """ Compute least squares cost of all splits of X
    
    Args:
        X np.array, shape n

    returns:
        arrarys of sum of squares error from left and right
    """
    pref_sum = np.cumsum(X)
    pref_sum_squares = np.cumsum(X**2)
    n = len(X)
    sum_squares_left = np.zeros(n+1)
    sum_squares_right = np.zeros(n+1)
    for i in range(1, n+1):
        mean_left = pref_sum[i-1]/i 
        mean_right = (pref_sum[-1] - pref_sum[i-1])/(n-i) # bad for i = n overwritten later
        sum_squares_left[i] = pref_sum_squares[i-1] + i * mean_left**2 - 2 * mean_left * pref_sum[i-1]        
        sum_squares_right[i] = (pref_sum_squares[-1] - pref_sum_squares [i-1]) + mean_right**2 * (n-i) - 2 * mean_right *(pref_sum[-1] - pref_sum[i-1])
    sum_squares_left[0] = 0
    sum_squares_right[0] = sum_squares_left[-1]
    sum_squares_right[-1] = 0
    return sum_squares_left, sum_squares_right
### END CODE


def main():
    """ Simple method testing """
    boston = load_boston()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                        boston.target,
                                                        test_size=0.2)

    baseline_accuracy = np.mean((y_test-np.mean(y_train))**2)
    print('Least Squares Cost of learning mean of training data:', baseline_accuracy) 
    print('Lets see if we can do better with just one question')
    D = RegressionStump()
    D.fit(X_train, y_train)
    print('idx, val, left, right', D.idx, D.val, D.left, D.right)
    print('Feature name of idx', boston.feature_names[D.idx])
    print('Score of model', D.score(X_test, y_test))
    print('lets compare with sklearn decision tree')
    dc = DecisionTreeRegressor(max_depth=1)
    dc.fit(X_train, y_train)
    dc_score = ((dc.predict(X_test)-y_test)**2).mean()
    print('dc score', dc_score)
    print('feature names - for comparison', list(enumerate(boston.feature_names)))
    plot_tree(dc, boston.feature_names)

if __name__ == '__main__':
    main()
