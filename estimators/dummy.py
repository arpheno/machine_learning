from sklearn.dummy import DummyClassifier, DummyRegressor

DummyClassifier.hyperparameters = {'strategy': ['most_frequent', 'prior', 'stratified', 'uniform', 'most_frequent']}
DummyRegressor.hyperparameters = {'strategy': ['mean', 'median', 'mean', 'median', 'mean']}
