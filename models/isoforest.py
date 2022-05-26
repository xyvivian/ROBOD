import os
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


class IsoForest():
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, **kwargs):
        # initialize
        self.isoForest = None
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.initialize_isoForest(**kwargs)


    def initialize_isoForest(self, seed=0, **kwargs):
        self.isoForest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                         contamination=self.contamination, n_jobs=-1, random_state=seed, **kwargs)

    def fit(self, train_X):
        print("Starting training...")
        start_time = time.time()
        self.isoForest.fit(train_X.astype(np.float32))
        end_time = time.time() - start_time
        return end_time

    def predict(self,test_X, test_y):
        print("Starting prediction...")
        scores = (-1.0) * self.isoForest.decision_function(test_X.astype(np.float32))  # compute anomaly score
        #y_pred = (self.isoForest.predict(test_X.astype(np.float32)) == -1) * 1  # get prediction
        auc = roc_auc_score(test_y, scores.flatten())
        print("AUCROC: %.4f" % auc)
        return scores
        