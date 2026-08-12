# y_pred = list of probability
# y_true = list of actual labels (true / false, or 1 / 0)

def safe_div(a, b):
    return None if b == 0 else (a / b)

def roc_curve(y_true, y_pred):
    # return a list of tuples (threshold, fpr, tpr) in ascending order by threshold
    # need to handle the same y_pred scenario

    if not y_pred:
        return []

    last_pred = None
    positive = 0
    negative = 0
    threshold = []
    fpr = []
    tpr = []

    for prediction, label in sorted(zip(y_pred, y_true), reverse=True):
        if last_pred != None and last_pred != prediction: # handles same prediction scenario
            threshold.append(last_pred)
            fpr.append(negative)
            tpr.append(positive)

        if label:
            positive += 1
        else:
            negative += 1

        last_pred = prediction
    
    # last data point
    threshold.append(last_pred)
    fpr.append(negative)
    tpr.append(positive)

    output = [(threshold, safe_div(fp, negative), safe_div(tp, positive)) for threshold, fp, tp in sorted(zip(threshold, fpr, tpr))]

    return output

def roc_auc(roc_curve):
    # roc_curve is a list of tuples (threshold, fpr, tpr)
    # use trapezoid method

    # sort by fpr
    roc_curve = sorted(roc_curve, key=lambda x: x[1])

    auc = 0
    pre_fpr = roc_curve[0][1]
    pre_tpr = roc_curve[0][2]

    for _, fpr, tpr in roc_curve[1:]:
        auc += (fpr - pre_fpr) * (tpr - pre_tpr) / 2         # triangle
        auc += (fpr - pre_fpr) * pre_tpr                     # square

        pre_fpr = fpr
        pre_tpr = tpr

    return auc


y_pred = [0.1, 0.2, 0.8, 0.8, 0.8, 0.9]
y_true = [False, True, False, True, True, True]

my_curve = roc_curve(y_true, y_pred)
my_auc = roc_auc(my_curve)
print(my_curve)
print(my_auc)

from sklearn.metrics import roc_auc_score
sk_auc = roc_auc_score(y_true, y_pred)

assert sk_auc == my_auc
print(f"PASS: auc = {my_auc:.3f}")
