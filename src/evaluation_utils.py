from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, auc, precision_score

def get_tpr_specific_fpr(fpr, tpr, s_fpr=0.01):
    for i, fp in enumerate(fpr):
        if fp > s_fpr:
            return fpr[i-1], tpr[i-1]
        

def print_evaluation_metrics(y_test, y_test_prediction, y_test_prob, model_name):
    print("accuracy_score", "for", model_name, accuracy_score(y_test, y_test_prediction))
    print("FA", "for", model_name, 1 - recall_score(y_test, y_test_prediction, pos_label=0))
    print("Detection rate i.e. recall_score", "for", model_name, recall_score(y_test, y_test_prediction))
    print("AUC", "for", model_name, roc_auc_score(y_test, y_test_prob))

    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    print("TPR@FPR=0.001", "for", model_name, get_tpr_specific_fpr(fpr, tpr, s_fpr=0.001))
    print("TPR@FPR=0.01", "for", model_name, get_tpr_specific_fpr(fpr, tpr, s_fpr=0.01))
    print("TPR@FPR=0.1", "for", model_name, get_tpr_specific_fpr(fpr, tpr, s_fpr=0.1))


def plot_roc_curve(y_test, y_test_prob, path_prefix, model_name='', max_fp=0.1):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=model_name + ' (AUC = %0.3f)' % auc(fpr, tpr))
    plt.xlim([0.0, max_fp])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(path_prefix +  "_ROC Curve", bbox_inches='tight')
    plt.show()
    

def plot_roc_curve_multiple(y_test, y_test_prob_list, path_prefix, model_names, max_fp=0.1):
    plt.figure()
    lw = 2
    for i, y_test_prob in enumerate(y_test_prob_list):
        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
        plt.plot(fpr, tpr, lw=lw, label=model_names[i] + ' (AUC = %0.3f)' % auc(fpr, tpr))
    plt.xlim([0.0, max_fp])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(path_prefix +  "_ROC Curve", bbox_inches='tight')
    plt.show()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_history_accuracy(history, epochs, path_prefix, sm=False, metrics=['acc','val_acc']):
    x = np.asarray(range(1, epochs + 1))
    # summarize history for accuracy
    plt.figure()
    if sm:
        plt.plot(x, smooth([y*100 for y in history[metrics[0]]],2))
        plt.plot(x, smooth([y*100 for y in history[metrics[1]]],2))
    else:
        plt.plot(x, [y*100 for y in history[metrics[0]]])
        plt.plot(x, [y*100 for y in history[metrics[1]]])

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    # plt.ylim(70,100) ###########################
    plt.legend(['Training', 'Test'], loc='lower right') #loc='lower right')
    plt.grid()
    fname = path_prefix + "_accuracy_history"
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    

def plot_history_loss(history, epochs, path_prefix, sm=False, metrics=['loss','val_loss']):
    x = np.asarray(range(1, epochs + 1))
    # summarize history for accuracy
    plt.figure()
    if sm:
        plt.plot(x, smooth([y*100 for y in history[metrics[0]]],2))
        plt.plot(x, smooth([y*100 for y in history[metrics[1]]],2))
    else:
        plt.plot(x, [y*100 for y in history[metrics[0]]])
        plt.plot(x, [y*100 for y in history[metrics[1]]])

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    # plt.ylim(70,100) ###########################
    plt.legend(['Training', 'Test'], loc='upper right') #loc='lower right')
    plt.grid()
    fname = path_prefix + "_loss_history"
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          fname='Confusion matrix', title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None:
        plt.title(title)
    cbar = plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.ylim(-0.5, 1.5)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            cbar.set_ticks([0, .2, .4, 0.6, 0.8, 1])
            cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        else:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(fname + ".png", bbox_inches='tight')


def compute_confusion_matrix(y_test, y_test_prediction, class_names, path_prefix):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_test_prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          fname=path_prefix + "_" + 'Confusion_matrix_without_normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          fname=path_prefix + "_" + 'Normalized_confusion_matrix')

    plt.show()