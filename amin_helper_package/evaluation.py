import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay


def comptue_error(y, y_hat):
    """
    Compute the error between the ground truth and the prediction.
    """
    return np.mean(np.abs(y_hat - y))


def to_categorial(y, y_hat, n_classes=4):
    y_c = np.array(y * n_classes).astype('int')
    y_c[y_c == n_classes] = n_classes - 1

    y_hat_c = np.array(y_hat * n_classes).astype('int')
    y_hat_c[y_hat_c == n_classes] = n_classes - 1
    return y_c, y_hat_c


def psuedo_fscore(y, y_hat, n_classes=4):
    y_c, y_hat_c = to_categorial(y, y_hat, n_classes)
    return f1_score(y_c, y_hat_c, average='macro')


def plot_predictions(y, y_hat, ax=None):
    if len(y) > 5000:
        ind = np.random.choice(len(y), 5000, replace=False)
        sns.regplot(x=y[ind], y=y_hat[ind], ax=ax)
    else:
        sns.regplot(x=y, y=y_hat, ax=ax)


def plot_confusion(y, y_hat, n_classes=4, ax=None):
    y_c, y_hat_c = to_categorial(y, y_hat, n_classes)
    ConfusionMatrixDisplay.from_predictions(y_c, y_hat_c, ax=ax)


def evaluate(y, y_hat, n_classes=4):
    print('Error: {:.2f}'.format(comptue_error(y, y_hat)))
    print('F1 score: {:.2f}'.format(psuedo_fscore(y, y_hat, n_classes)))


def evaluate_plots(y, y_hat, n_classes=4):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].set_title('Ground truth vs. prediction')
    plot_predictions(y, y_hat, ax=ax[0])

    ax[1].set_title('Pseudo confusion matrix')
    plot_confusion(y, y_hat, n_classes=n_classes, ax=ax[1])
    plt.show()
