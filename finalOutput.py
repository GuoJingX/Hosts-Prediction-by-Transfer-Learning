import pandas as pd
import numpy as np
from collections import Counter

def final(acc, pred, prob, T):
    weights = list(acc.values()) / sum(acc.values())

    for i in range(T):

        if i == 0:
            prob_ = prob[i]
            prob_ = prob_[np.newaxis, :]

        else:
            prob_new = prob[i]
            prob_new = prob_new[np.newaxis, :]
            prob_ = np.vstack((prob_, prob_new))

    prob_final = prob_.mean(axis=0)

    pred_dataframe = pd.DataFrame(pred)
    len_pred = len(pred_dataframe)
    pred_final = []

    for i in range(len_pred):
        labelall = pred_dataframe.iloc[i]
        a = Counter(labelall)
        lena = len(a)
        b = Counter(labelall).most_common(1)
        label = b[0][0]
        pred_final.append(label)
    return pred_final, prob_final
