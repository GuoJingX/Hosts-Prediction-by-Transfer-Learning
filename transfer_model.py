import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn
from array import array
from sklearn.preprocessing import label_binarize
from sklearn import neural_network
from sklearn import tree
import pandas as pd
from resample import fit_sample
from AUC import AUC


class TriTransferLearning:
    def __init__(self, T):
        '''
        :param T: iteration num
        '''
        self.T = T

    def fit(self, src_X, src_Y, tar_train_X, tar_train_Y, tar_test_X, tar_test_y):
        pred = {}
        prob = {}
        acc = {}
        auc = {}
        for t in range(self.T):
            src_Xnew, src_Ynew = fit_sample(src_X, src_Y)  # undersampling
            train_x = np.vstack((src_Xnew, tar_train_X))
            train_y = np.vstack((src_Ynew, tar_train_Y))

            model_LR = sklearn.linear_model.LogisticRegression(penalty='l2', C=10, solver='sag', multi_class='auto', max_iter=5000)
            model_LR.fit(train_x, train_y.ravel())
            y_pred_LR = model_LR.predict(tar_test_X)

            model_SVM = sklearn.svm.SVC(C=10, kernel='sigmoid', gamma=10)
            model_SVM.fit(train_x, train_y.ravel())
            y_pred_SVM = model_SVM.predict(tar_test_X)

            tar_hc_x = tar_test_X[y_pred_LR == y_pred_SVM]
            tar_hc_y = y_pred_LR[y_pred_LR == y_pred_SVM]

            train_x_NN = np.vstack((train_x, tar_hc_x))
            train_y_NN = np.vstack((train_y, tar_hc_y.reshape(-1, 1)))

            # Neural Network
            model_NN = neural_network.MLPClassifier(alpha=0.01, max_iter=10000, solver='adam')
            model_NN.fit(train_x_NN, train_y_NN.ravel())

            y_pred_NN = model_NN.predict(tar_test_X)
            y_pred_Pro = model_NN.predict_proba(tar_test_X)
            acc_NN_test = sklearn.metrics.accuracy_score(y_pred_NN, tar_test_y)

            y_pred_train = model_NN.predict(tar_train_X)
            acc_NN_train = sklearn.metrics.accuracy_score(y_pred_train, tar_train_Y)

            acc[t] = acc_NN_train
            pred[t] = y_pred_NN
            prob[t] = y_pred_Pro


        return acc, pred, prob










