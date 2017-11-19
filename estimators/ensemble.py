from lightgbm import LGBMRegressor, LGBMClassifier
from scipy.stats import expon, uniform
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from xgboost import XGBRegressor, XGBClassifier

base = {
    'learning_rate': expon(0, 0.1),
    'n_estimators': [100, 300, 700, 1200],
}
AdaBoostRegressor.hyperparameters = {**base}
AdaBoostClassifier.hyperparameters = {**base}

xgb_base = {
    **base,
    'min_child_weight': [1, 3, 5],
    'gamma': expon(0, 0.3),
    'max_depth': [7, 9, 11],
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
}

XGBRegressor.hyperparameters = {**xgb_base}
XGBClassifier.hyperparameters = {**xgb_base}

lgbm_base = {
    **base,
    'n_estimators': [8, 16, 24],
    'num_leaves': [10, 30, 50, 100],
    'colsample_bytree': uniform(0.5, 0.5),
    'subsample': uniform(0.5, 0.5),
    'reg_alpha': expon(1, 0.1),
    'reg_lambda': expon(1, 0.1),
    'min_child_weight': [0, 3, 5, 8],
}
LGBMRegressor.hyperparameters = {**lgbm_base}
LGBMClassifier.hyperparameters = {**lgbm_base}

gbm_base = {
    **base,
    'criterion': ['mse', 'mae'],
    'max_features': ['log2', 'sqrt', 'auto', None],
}
GradientBoostingRegressor.hyperparameters = {**gbm_base, 'loss': ['ls', 'lad', 'huber', 'quantile']}
GradientBoostingClassifier.hyperparameters = {**gbm_base, 'loss': ['exponential', 'deviance'], }

# Random forests don't actually have learning rates
random_forest_base = {
    'max_features': ['log2', 'sqrt', 'auto', None],
    'n_estimators': [100, 300, 700, 1200],
}
RandomForestRegressor.hyperparameters = {**random_forest_base, 'criterion': ['mse', 'mae'], }
RandomForestClassifier.hyperparameters = {**random_forest_base, 'criterion': ["gini", 'entropy'], }
