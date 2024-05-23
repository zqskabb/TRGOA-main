import sklearn, warnings, torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np

def filter_sample(pred_label, true_label):
    indices_to_keep = []
    for i in range(true_label.shape[0]):
        if not torch.all(true_label[i, :] == 0):
            indices_to_keep.append(i)
    true_label_filtered = true_label[indices_to_keep]
    pred_label_filtered = pred_label[indices_to_keep]
    return pred_label_filtered, true_label_filtered

def auprc(ytrue, ypred):
  p, r, t =  precision_recall_curve(ytrue, ypred)
  return auc(r,p)

def Fmax_AUPRC(ypred1, ytrue1):
	fmax = 0
	prec_list = []
	recall_list=[]
	ytrue=[]
	ypred=[]	
	for i in range(len(ytrue1)):
		if np.sum(ytrue1[i]) >0:
			ytrue.append(ytrue1[i])
			ypred.append(ypred1[i])	
	for t in range(1, 101):
		thres = t/100.
		thres_array=np.ones((len(ytrue), len(ytrue[0])), dtype=np.float32) * thres
		pred_labels = np.greater(ypred, thres_array).astype(int)
		tp_matrix =pred_labels*ytrue
		tp = np.sum(tp_matrix, axis=1, dtype=np.int32)
		tpfp = np.sum(pred_labels, axis=1)
		tpfn = np.sum(ytrue,axis=1)
		avgprs=[]
		for i in range(len(tp)):
			if tpfp[i]!=0:
				avgprs.append(tp[i]/float(tpfp[i]))
		if len(avgprs)==0:
			continue
		avgpr = np.mean(avgprs)
		avgrc = np.mean(tp/tpfn)
		prec_list.append(avgpr)
		recall_list.append(avgrc)
		f1 = 2*avgpr*avgrc/(avgpr+avgrc)
		fmax=max(fmax, f1)
	return fmax, auprc(np.array(ytrue).flatten(), np.array(ypred).flatten())

def f_score(pred, true):
    ytrue=[]
    ypred=[]
    for i in range(len(true)):
        if np.sum(true[i]) >0:
            ytrue.append(true[i])
            ypred.append(pred[i])	
    mt = 0
    precision = 0
    recall = 0
    pr = 0
    rc = 0
    N = 0
    f = 0
    for i, predicted in enumerate(pred):
        gt = true[i]
        if sum(gt) == 0:
            continue
        tp = sum(np.logical_and(gt, predicted).astype(int))
        fp = sum(predicted) - tp
        fn = sum(gt) - tp
        N += 1
        recall = tp / (1.0*(tp + fn))
        rc += recall
        if np.sum(predicted) >0:
            mt+=1
            precision = tp / (1.0*(tp + fp))
            pr += precision       
    rc /= N 
    if mt > 0:
        pr /= mt  
    if pr + rc > 0:
        f = 2 * pr * rc / (pr + rc)
    return f, auprc(np.array(ytrue).flatten(), np.array(ypred).flatten())

def f_score_1(preds, labels):
    ytrue=[]
    ypred=[]
    for i in range(len(labels)):
        if np.sum(labels[i]) >0:
            ytrue.append(labels[i])
            ypred.append(preds[i])	

    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, auprc(np.array(ytrue).flatten(), np.array(ypred).flatten())


def fmax_aupr(Ytrue, Ypred, nrThresholds):
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ff = np.zeros(thresholds.shape)
    pr = np.zeros(thresholds.shape)
    rc = np.zeros(thresholds.shape)
    for i, t in enumerate(thresholds):
        thr = np.round(t, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr[i], rc[i], ff[i], _ = metrics.precision_recall_fscore_support(Ytrue.astype(int), (Ypred >=t).astype(int), average='samples')
    AUPRC = metrics.average_precision_score(Ytrue, Ypred, average='samples')
    return np.max(ff), AUPRC

a = 489
b = 2432
def count_evaluation_GOGAT(pred_y, true_y):
    mf_pred_y, mf_true_y = filter_sample(pred_y[:, :a], true_y[:, :a])
    bp_pred_y, bp_true_y = filter_sample(pred_y[:, a:b], true_y[:, a:b])
    cc_pred_y, cc_true_y = filter_sample(pred_y[:, b:], true_y[:, b:])
    mf_Fmax, mf_AUPRC = f_score_1(mf_pred_y.numpy(), mf_true_y.numpy())
    bp_Fmax, bp_AUPRC = f_score_1(bp_pred_y.numpy(), bp_true_y.numpy())
    cc_Fmax, cc_AUPRC = f_score_1(cc_pred_y.numpy(), cc_true_y.numpy())
    result = (mf_Fmax, mf_AUPRC, bp_Fmax, bp_AUPRC, cc_Fmax, cc_AUPRC)
    return result
def count_evaluation_TALE(pred_y, true_y):
    mf_pred_y, mf_true_y = filter_sample(pred_y[:, :a], true_y[:, :a])
    bp_pred_y, bp_true_y = filter_sample(pred_y[:, a:b], true_y[:, a:b])
    cc_pred_y, cc_true_y = filter_sample(pred_y[:, b:], true_y[:, b:])
    mf_Fmax, mf_AUPRC = Fmax_AUPRC(mf_pred_y.numpy(), mf_true_y.numpy())
    bp_Fmax, bp_AUPRC = Fmax_AUPRC(bp_pred_y.numpy(), bp_true_y.numpy())
    cc_Fmax, cc_AUPRC = Fmax_AUPRC(cc_pred_y.numpy(), cc_true_y.numpy())
    result = (mf_Fmax, mf_AUPRC, bp_Fmax, bp_AUPRC, cc_Fmax, cc_AUPRC)
    return result
def count_evaluation_TALE_1(pred_y, true_y):
    mf_Fmax, mf_AUPRC = Fmax_AUPRC(pred_y[:, :a].numpy(), true_y[:, :a].numpy())
    bp_Fmax, bp_AUPRC = Fmax_AUPRC(pred_y[:, a:b].numpy(), true_y[:, a:b].numpy())
    cc_Fmax, cc_AUPRC = Fmax_AUPRC(pred_y[:, b:].numpy(), true_y[:, b:].numpy())
    result = (mf_Fmax, mf_AUPRC, bp_Fmax, bp_AUPRC, cc_Fmax, cc_AUPRC)
    return result

def count_evaluation_HEAL(pred_y, true_y, nrThresholds):
    mf_pred_y, mf_true_y = filter_sample(pred_y[:, :a], true_y[:, :a])
    bp_pred_y, bp_true_y = filter_sample(pred_y[:, a:b], true_y[:, a:b])
    cc_pred_y, cc_true_y = filter_sample(pred_y[:, b:], true_y[:, b:])
    mf_Fmax, mf_AUPRC = fmax_aupr(mf_true_y.numpy(), mf_pred_y.numpy(), nrThresholds)
    bp_Fmax, bp_AUPRC = fmax_aupr(bp_true_y.numpy(), bp_pred_y.numpy(), nrThresholds)
    cc_Fmax, cc_AUPRC = fmax_aupr(cc_true_y.numpy(), cc_pred_y.numpy(), nrThresholds)
    result = (mf_Fmax, mf_AUPRC, bp_Fmax, bp_AUPRC, cc_Fmax, cc_AUPRC)
    return result


def count_evaluation(pred_y, true_y):
    pred_y, true_y = filter_sample(pred_y, true_y)
    cc_Fmax, cc_AUPRC = Fmax_AUPRC(pred_y.numpy(), true_y.numpy())
    result = (cc_Fmax, cc_AUPRC)
    return result