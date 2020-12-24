import numpy as np
from sklearn.utils import check_random_state


def fit_sample(X, y):
    """Resample the dataset.
    """
    label = np.unique(y)
    stats_c_ = {}
    min_n = 10000000
    for i in label:
        nk = sum(y == i)
        stats_c_[i] = nk
        if nk < min_n:
            min_n = nk
            min_c_ = i

    X_resampled = X[y[:,0] == min_c_]
    y_resampled = y[y[:,0] == min_c_]

    for key in stats_c_.keys():

        # If this is the min class, skip it
        if key == min_c_:
            continue

        # Define the number of sample to create
        num_samples = int(stats_c_[key]-stats_c_[min_c_])
        # Pick some elements at random
        random_state = check_random_state(None)
        # indx = random_state.randint(low=0, high=stats_c_[key], size=num_samples)
        indx = random_state.randint(low=0, high=stats_c_[key], size=min_n)
        # Concatenate to the majority class
        X_resampled = np.concatenate((X_resampled, X[y[:,0] == key][indx]), axis=0)
        # X_resampled = vstack([X_resampled, X[y == key], X[y == key][indx]])
        y_resampled = np.concatenate((y_resampled, y[y[:,0] == key][indx]), axis=0)

    return X_resampled, y_resampled


