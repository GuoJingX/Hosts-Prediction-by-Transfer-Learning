import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from transfer_model import TriTransferLearning
from finalOutput import final
from AUC import AUC
from sklearn.preprocessing import LabelEncoder


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self, Xs, Ys, Xt_train, Yt_train, Xt_test, Yt_test):

        Xt = np.vstack((Xt_train, Xt_test))
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0

        Y_tar_pseudo = None

        for t in range(self.T):

            N = 0
            M0 = e * e.T * C

            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                Y_tar_pseudo = Y_tar_pseudo.ravel()
                for c in range(0, C):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    a = len(Ys[np.where(Ys == c)])
                    e[np.where(tt == True)] = 1 / a
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            Xt_new_train = Xt_new[:len(Yt_train), :]
            Xt_new_test = Xt_new[len(Yt_train):, :]

            clf = KNeighborsClassifier(n_neighbors=1)
            knn_X = np.vstack((Xs_new, Xt_new_train))
            knn_Y = np.vstack((Ys, Yt_train))
            clf.fit(knn_X, knn_Y.ravel())
            Y_pseudo_test = clf.predict(Xt_new_test)
            Y_pseudo_test = Y_pseudo_test.reshape(-1, 1)
            Y_tar_pseudo = np.concatenate((Yt_train, Y_pseudo_test), axis=0)

        return Xs_new, Xt_new_train, Xt_new_test


if __name__ == '__main__':
    # Source
    SourceData = pd.read_csv('data/finally/Source.csv').values
    Source_X = np.array(SourceData[:,1:4230],dtype='float64')
    Source_Y = SourceData[:,4230]

    le = LabelEncoder()
    le.fit_transform(Source_Y)
    Source_Y = le.transform(Source_Y)

    # Target
    CoronaData = pd.read_csv('data/finally/Coronavirus.csv').values
    auc_HTL_, fpr_HTL_, tpr_HTL_, acc_HTL_ = [], [], [], []

    np.random.shuffle(CoronaData)
    target_X = np.array(CoronaData[:, 1:4230], dtype='float64')
    target_Y = CoronaData[:, 4230]
    tar_trainlist = []
    tar_testlist = []
    labels = np.unique(target_Y)
    for key in range(labels.shape[0]):
        index = target_Y == labels[key]

        t1 = CoronaData[index]
        for j in range(t1.shape[0]):
            if j < 3:
                tar_trainlist.append(t1[j])
            else:
                tar_testlist.append(t1[j])

    tar_train = np.array(tar_trainlist)
    tar_test = np.array(tar_testlist)
    tar_train_x = np.array(tar_train[:, 1:4230], dtype='float64')
    tar_train_Y = np.array(tar_train[:, 4230])
    tar_train_Y = le.transform(tar_train_Y)
    tar_test_x = np.array(tar_test[:, 1:4230], dtype='float64')
    tar_test_Y = np.array(tar_test[:, 4230])
    tar_test_Y = le.transform(tar_test_Y)

    Source_Y = Source_Y.reshape(-1, 1)
    tar_train_Y = tar_train_Y.reshape(-1, 1)
    tar_test_Y = tar_test_Y.reshape(-1, 1)

    jda = JDA(kernel_type='linear', dim=200, lamb=0.1, gamma=0.01, T=10)
    Xs_new_JDA, Xt_new_train, Xt_new_test = jda.fit_predict(Source_X, Source_Y, tar_train_x, tar_train_Y,
                                                            tar_test_x, tar_test_Y)
    iteration = 10
    TransModel = TriTransferLearning(T=iteration)

    acc, pred, prob = TransModel.fit(Xs_new_JDA, Source_Y, Xt_new_train, tar_train_Y,
                                     Xt_new_test, tar_test_Y)
    pred_final, prob_final = final(acc, pred, prob, iteration)
    pred_final = np.array(pred_final)
    acc_HTL = sklearn.metrics.accuracy_score(pred_final, tar_test_Y)
    auc_HTL, fpr_HTL, tpr_HTL = AUC(tar_test_Y, prob_final)

    print("ACC:", acc_HTL)
    print("AUC", auc_HTL)

