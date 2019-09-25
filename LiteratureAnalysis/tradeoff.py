import pickle
from tools import load_model
import numpy as np
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


plt.rc('font', family='Times New Roman')
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('axes', labelsize=36)

THETAS = np.linspace(0, 1, 11)


def merge(y_1, y_2):
    y_1 = y_1[:, np.newaxis]
    y_2 = y_2[:, np.newaxis]
    return np.hstack((y_1, y_2))


def find_optimal(data):
    index = np.argmax(data)
    index_mid = np.where(THETAS==0.5)[0][0]
    if data[index] > data[index_mid]:
        return THETAS[index]
    else:
        return THETAS[index_mid]


def evaluate(y, proba):
    y = np.array(y)
    proba = np.array(proba)
    values = []
    for theta in THETAS:
        proba_new = np.where(proba >= theta, 1, 0)
        f1 = f1_score(y, proba_new)
        values.append(f1)
    return np.array(values)

if __name__ =="__main__":
    y_test, pred_proba = load_model("data/Data_flow/tradeoff")
    pred_label = np.where(pred_proba > 0.5, 1,0)
    # y_test, test_proba = load_model("data/Data_flow/tradeoff-backup")

    print("performance before setting threshold")
    print(classification_report(y_test[:, :-1], pred_label[:, :-1]))

    F = y_test[:, 0]
    proba_F = pred_proba[:, 0]

    I = y_test[:, 1]
    proba_I = pred_proba[:, 1]

    # calculate f1 scores
    values_F = evaluate(F, proba_F)
    values_I = evaluate(I, proba_I)

    # find the optimal theta
    theta_F = find_optimal(values_F)
    theta_I = find_optimal(values_I)

    # assign 1 or 0 to probabilities
    F_pred = np.where(proba_F > theta_F, 1, 0)
    I_pred = np.where(proba_I > theta_I, 1, 0)

    # merge F and I labels
    y_true = merge(F, I)
    y_pred = merge(F_pred, I_pred)

    print("performance after setting threshold")
    print(classification_report(y_true, y_pred))

    print("the best threshold of  financial flow is: %.2f" % theta_F)
    print("the best threshold of  information flow is: %.2f" % theta_I)

    plt.plot(THETAS, values_I, 'b-.',lw=3, marker='*', ms=15, label='Information flow')
    plt.plot(THETAS, values_F,  'r-.', marker='o', ms=15, lw=3, label='Financial flow')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(THETAS)
    plt.legend(loc='upper right', ncol=1, prop={'size': 20})
    plt.xlabel(r'Thresholds $\theta$')
    plt.ylabel('F1 score')
    plt.grid(True, linestyle="--", linewidth=1, color='black')

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.show()
