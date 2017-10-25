from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score

from genetic.estimator_parameters import params
from genetic.parameter_optimization import tune_params_genetic
from swozny_ml import benchmark_models, tune_params

# Config
filename = 'data.csv'
index = 'shot_id'
target = 'shot_made_flag'
prediction_params = ['Angle', 'Distance']
scoring = 'accuracy'
voting = 'soft'

# Read data
data = pd.read_csv(filename, index_col=index)
print(data.head())

# Prepare data
training_cond = data[target].notnull()
X_train = data[training_cond][prediction_params]
y_train = data[training_cond][target]

# Benchmark different models
classifiers = [estimator for estimator in params if "Classifier" in estimator.__name__] + [LogisticRegression]
benchmark = benchmark_models(classifiers, X_train, y_train, scoring='accuracy')
considered_algorithms = benchmark.sort_values('Mean').tail(5)

# Report
plot_benchmark(benchmark)
print(considered_algorithms)

# Tune the model
if True:
    tuned = tune_params(considered_algorithms['Algorithm'], X_train, y_train, scoring='accuracy')
else:
    tuned = tune_params_genetic(considered_algorithms['Algorithm'], X_train, y_train, scoring=scoring)

# Calibrate
calibrated = [CalibratedClassifierCV(model).fit(X_train, y_train) for model in tuned]

# Correlation
predictions = pd.concat(
    [pd.Series(model.predict(X_train), name=type(model.base_estimator).__name__) for model in calibrated], axis=1)
sns.heatmap(predictions.corr())

# Ensemble
survival_model = VotingClassifier([(type(model.base_estimator).__name__, model) for model in calibrated], voting=voting)
survival_model.fit(X_train, y_train)

# Predict
y_pred = cross_val_predict(survival_model, X=X_train, y=y_train)
score = cross_val_score(survival_model, X=X_train, y=y_train)
print(f"Final cross validation score is {score}")

# Confusion
sns.heatmap(confusion_matrix(y_pred, y_train), annot=True)

# Generate output
test_cond = ~training_cond
X_test = data[test_cond][prediction_params]
y_pred = survival_model.predict_proba(X_test)
X_test['shot_made_flag'] = y_pred[:, 1]
X_test.shot_made_flag.to_csv('pred_kobe.csv', header=True)
