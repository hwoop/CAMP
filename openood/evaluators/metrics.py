import numpy as np
from sklearn import metrics


def compute_all_metrics(
    conf,
    label,
    pred,
):

    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf=conf, label=label, tpr_th=recall)

    accuracy = acc(pred=pred, label=label)

    results = [fpr, auroc, aupr_in, aupr_out, accuracy]

    return results


def acc(
    pred,
    label,
):

    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


def auc_and_fpr_recall(
    conf,
    label,
    tpr_th,
):

    # OpenOOD convention: OOD is the positive class (FPR@95 = share of ID called OOD at 95% OOD recall)
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # postprocessors score ID higher, hence -conf for the OOD-positive curves
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr
