from scipy.stats import expon
from sklearn.svm import SVR, SVC

SVR.hyperparameters = {
    'C': expon(0.8, 1),
    'kernel': ['rbf', 'linear'],
}
SVC.hyperparameters = {
    'C': expon(0.8, 0.5),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
}
