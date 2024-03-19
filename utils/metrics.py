import numpy as np
from sklearn.metrics import f1_score



def compute_EXP_F1(pred,target):
    pred_labels = []
    pred = np.array(pred)
    target = np.array(target)
    for i in range(pred.shape[0]):
        l = np.argmax(pred[i])
        pred_labels.append(l)
        
    F1s = f1_score(target, pred_labels, average=None)
    macro_f1 = np.mean(F1s)
    return F1s,macro_f1



