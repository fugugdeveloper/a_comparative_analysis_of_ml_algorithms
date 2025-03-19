from itertools import cycle
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error,
                             mean_absolute_percentage_error, log_loss, auc, roc_curve, precision_recall_curve,
                             make_scorer, classification_report, )
import logging
import time
import numpy as np
from sklearn.preprocessing import label_binarize

def mean_bias_deviation(y_true, y_pred):
    return np.mean(y_pred - y_true)
def relative_squared_error(y_true, y_pred):
    return np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_true - np.mean(y_true)))
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = np.square(error) / 2
    linear_loss = delta * (np.abs(error) - delta / 2)
    return np.where(is_small_error, squared_loss, linear_loss).mean()
# Function to evaluate regression models
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    mbd = mean_bias_deviation(y_test, y_pred)
    rse = relative_squared_error(y_test, y_pred)
    huber = huber_loss(y_test, y_pred)

    logging.info(f"Model: {model.__class__.__name__} (Regression)")
    logging.info(f"MAE: {mae}")
    logging.info(f"MSE: {mse}")
    logging.info(f"RMSE: {rmse}")
    logging.info(f"R²: {r2}")
    logging.info(f"Adjusted R²: {adjusted_r2}")
    logging.info(f"MAPE: {mape}")
    logging.info(f"Explained Variance Score: {evs}")
    logging.info(f"Max Error: {max_err}")
    logging.info(f"Mean Bias Deviation: {mbd}")
    logging.info(f"Relative Squared Error: {rse}")
    logging.info(f"Huber Loss: {huber}")
    logging.info(f"Training Time: {training_time}s")
    logging.info(f"Prediction Time: {prediction_time}s")

    return {
        'model': model_name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'adjusted_r2': adjusted_r2,
        'mape': mape,
        'evs': evs,
        'max_err': max_err,
        'mbd': mbd,
        'rse': rse,
        'huber': huber,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'y_test': y_test,
    }


# Function to evaluate classification models
def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    if len(np.unique(y_test)) > 2:
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        roc_auc = roc_auc_score(y_test_binarized, model.predict_proba(X_test), multi_class='ovo', average='weighted')
    else:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    logloss = log_loss(y_test, model.predict_proba(X_test))

    precisions, recalls, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    pr_auc = auc(recalls, precisions)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Model: {model.__class__.__name__} (Classification)")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC-AUC: {roc_auc}")
    logging.info(f"Log Loss: {logloss}")
    logging.info(f"PR-AUC: {pr_auc}")
    logging.info(f"Specificity: {specificity}")
    logging.info(f"Confusion Matrix:\n {cm}")
    logging.info(f"Training Time: {training_time}s")
    logging.info(f"Prediction Time: {prediction_time}s")
    logging.info(f"Classification Report:\n {report}")

    return {
        'model': model.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'logloss': logloss,
        'pr_auc': pr_auc,
        'specificity': specificity,
        'cm': cm,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'y_test': y_test,
        'probs': model.predict_proba(X_test),
        'report': report
    }

def plot_classification_accuracy(models, accuracies):
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color='lightblue')
    plt.title('Accuracy of Different Classification Models')
    plt.xlabel('Classification Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(f'classification_accuracy.png')
    plt.close()


def evaluate_interpretability(model):
    if isinstance(model, RandomForestClassifier):
        return "Medium (feature importance available)"
    elif isinstance(model, SVC):
        return "Low (complex decision boundaries)"
    elif isinstance(model, MLPClassifier):
        return "Low (black-box model)"
    else:
        return "Unknown"

def plot_classification_report(report, model_name):
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="viridis")
    plt.title(f'Classification Report for {model_name}')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.savefig(f'classification_report_{model_name}.png')
    plt.close()
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()


def plot_roc_curve(y_test, probs, model_name):
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(set(y_test))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()


def plot_precision_recall_curve(y_test, probs, model_name):
    precision = {}
    recall = {}
    n_classes = len(set(y_test))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test, probs[:, i], pos_label=i)

    plt.figure()
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR curve of class {0}'
                       ''.format(i))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')
    plt.savefig(f'precision_recall_curve_{model_name}.png')
    plt.close()

def plot_regression_metrics_comparison(results):
    df = pd.DataFrame(results)
    expected_metrics = ['explained_variance', 'huber_loss', 'max_error',
                        'mean_absolute_error', 'mean_absolute_percentage_error',
                        'mean_bias_deviation', 'mean_squared_error',
                        'r2_score', 'relative_squared_error', 'root_mean_squared_error']

    # Check which metrics are actually in the DataFrame
    available_metrics = [metric for metric in expected_metrics if metric in df.columns]

    # Melt the DataFrame based on available metrics
    df_melted = df.melt(id_vars=['model'], value_vars=available_metrics)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x='variable', y='value', hue='model')
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()


def plot_classification_metrics_comparison(results):
    df = pd.DataFrame(results)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'training_time', 'prediction_time']
    df_melted = df.melt(id_vars=['model'], value_vars=metrics)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x='variable', y='value', hue='model')
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()


def plot_classification_learning_curve(estimator, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel(f'Training {model_name}')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(f'learning_curve_{model_name}.png')
    plt.close()


def create_classification_target_column(df):
    # Example: Weighted sum of different columns to create a target column
    df['custom_classification_target'] = (
            3.0 * df['is_bug_fix'] +
            2.0 * df['merge_status'] +
            4.0 * df['code_quality_rate']+
            4.0 * df['code_quality_issues']
    )
    return df

def create_regression_target_column(df):
    # Example: Weighted sum of different columns to create a target column
    df['custom_regression_target'] = (
            2.0 * df['files_changed'] +
            2.0 * df['lines_added'] +
            2.0 * df['lines_deleted'] +
            2.0 * df['time_to_merge'] +
            2.0 * df['num_commits'] +
            3.0 * df['num_comments'] +
            3.0 * df['num_reviewers'] +
            2.0 * df['author_experience'] +
            3.0 * df['review_time'] +
            3.0 * df['num_approvals'] +
            3.0 * df['review_sentiment_score']

    )
    return df
def plot_regression_learning_curve(model, X_train, y_train, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training error')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation error')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Learning Curves for {model_name}')
    plt.savefig(f'regression_learning_curve_{model_name}.png')
    plt.legend(loc='best')
    plt.show()

def plot_error_distribution(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Residuals for {model_name}')
    plt.savefig(f'error_distribution_{model_name}.png')
    plt.show()

def plot_prediction_vs_actual(y_test, y_pred, model_name):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Prediction vs Actual for {model_name}')
        plt.savefig(f'prediction_vs_actual{model_name}.png')
        plt.show()
def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model_name}')
    plt.savefig(f'residuals{model_name}.png')
    plt.show()


def run_regression_pipeline(X_train, X_test, y_train, y_test):
    models = [
        {'model': RandomForestRegressor(), 'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }},
        {'model': SVR(), 'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }},
        {'model': MLPRegressor(max_iter=1000), 'params': {
            'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd']
        }}
    ]

    results = []
    for m in models:
        logging.info(f"Tuning hyperparameters for {m['model'].__class__.__name__}...")
        grid_search = GridSearchCV(m['model'], m['params'], cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
                                   verbose=1)
        result = evaluate_regression_model(grid_search, X_train, X_test, y_train, y_test,m['model'].__class__.__name__)
        results.append(result)
        plot_residuals(result['y_test'], result['y_pred'], m['model'].__class__.__name__)
        plot_regression_learning_curve(m['model'], X_train,y_train, m['model'].__class__.__name__)
        plot_error_distribution(result['y_test'], result['y_pred'],  m['model'].__class__.__name__)
        plot_prediction_vs_actual(result['y_test'], result['y_pred'], m['model'].__class__.__name__)

        # base_learners = [
        #     ('rf', RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=5)),
        #     ('svm', SVR(C=1, kernel='rbf', gamma='scale')),
        #     ('mlp', MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, early_stopping=True))
        # ]

        # Final estimator
        # final_estimators = {
        #     'LinearRegression': LinearRegression(),
        #     'Ridge': Ridge(),
        #     'ElasticNet': ElasticNet(),
        #     'GradientBoostingRegressor': GradientBoostingRegressor()
        # }
        # scorer = make_scorer(mean_squared_error, greater_is_better=False)

        # Store results
        estimator_results = {}

        # Evaluate each final estimator
        # for name, estimator in final_estimators.items():
        #     stacking_regressor_estimator = StackingRegressor(estimators=base_learners, final_estimator=estimator, cv=5)
        #     scores = cross_val_score(stacking_regressor_estimator, X_train, y_train, cv=5, scoring=scorer)
        #     estimator_results[name] = (scores.mean(), scores.std())
        #     print(f"{name}: Mean MSE = {-scores.mean():.4f}, Std = {scores.std():.4f}")

        # Select the best final estimator
        # best_name = max(estimator_results, key=lambda k: estimator_results[k][0])
        # best_final_estimator = final_estimators[best_name]
        # print("best_final_estimator: ", best_final_estimator)
        # Stacking Regressor
        # stacking_regressor = StackingRegressor(
        #     estimators=base_learners,
        #     final_estimator=Ridge(alpha=1.0),
        #     cv=5,
        #     verbose=1
        # )
        # result = evaluate_regression_model(stacking_regressor, X_train, X_test, y_train, y_test, m['model'].__class__.__name__)
        # results.append(result)
        #
        # plot_residuals(result['y_test'], result['y_pred'], m['model'].__class__.__name__)
        # plot_prediction_vs_actual(result['y_test'], result['y_pred'],m['model'].__class__.__name__)
        # plot_error_distribution(result['y_test'], result['y_pred'], m['model'].__class__.__name__)
        # plot_prediction_vs_actual(result['y_test'], result['y_pred'], m['model'].__class__.__name__)
        # plot_residuals(result['y_test'], result['y_pred'], m['model'].__class__.__name__)
        # plot_regression_learning_curve(m['model'], X_train, y_train, m['model'].__class__.__name__)

    return results


#  for classification models
def run_classification_pipeline(X_train, X_test, y_train, y_test, X, y):
    models = [
        {'model': RandomForestClassifier(), 'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }},
        {'model': SVC(probability=True), 'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }},
        {'model':  MLPClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42), 'params': {
            'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd']
        }}
    ]

    results = []
    accuracies_grid = []
    accuracies_stack= []
    for m in models:
        logging.info(f"Tuning hyperparameters for {m['model'].__class__.__name__}...")
        grid_search = GridSearchCV(m['model'], m['params'], cv=5, n_jobs=-1, scoring='f1_weighted', verbose=1)
        result = evaluate_classification_model(grid_search, X_train, X_test, y_train, y_test )
        results.append(result)
        accuracies_grid.append(result['accuracy'])
        plot_classification_report(result['report'], m['model'].__class__.__name__)
        plot_confusion_matrix(confusion_matrix(y_test, result['y_pred']), m['model'].__class__.__name__)
        plot_roc_curve(result['y_test'], result['probs'], m['model'].__class__.__name__)
        plot_precision_recall_curve(result['y_test'], result['probs'], m['model'].__class__.__name__)
        plot_classification_learning_curve(m['model'], X, y, m['model'].__class__.__name__)

        # base_learners = [
        #     ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5)),
        #     ('svm', SVC(C=1, kernel='rbf', gamma='scale', probability=True)),
        #     ('mlp', MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000))
        # ]
        # final_estimators = {
        #     'LogisticRegression': LogisticRegression(),
        #     'RidgeClassifier': RidgeClassifier(),
        #     'GradientBoostingClassifier': GradientBoostingClassifier()
        # }
        # estimator_results = {}

        # Evaluate each final estimator
        # for name, estimator in final_estimators.items():
        #     stacking_clf_estimator = StackingClassifier(estimators=base_learners, final_estimator=estimator, cv=5)
        #     scores = cross_val_score(stacking_clf_estimator, X_train, y_train, cv=5, scoring='f1_weighted')
        #     estimator_results[name] = (scores.mean(), scores.std())
        #     print(f"{name}: Mean F1 Score = {scores.mean():.4f}, Std = {scores.std():.4f}")

        # Select the best final estimator
        # best_name = max(estimator_results, key=lambda k: results[k][0])
        # best_final_estimator = final_estimators[best_name]
        # stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
        # result = evaluate_classification_model(stacking_clf, X_train, X_test, y_train, y_test)
        # results.append(result)
        # accuracies_stack.append(result['accuracy'])
        # plot_classification_report(result['report'], m['model'].__class__.__name__)
        # plot_confusion_matrix(confusion_matrix(y_test, result['y_pred']), m['model'].__class__.__name__)
        # plot_roc_curve(result['y_test'], result['probs'], m['model'].__class__.__name__)
        # plot_precision_recall_curve(result['y_test'], result['probs'], m['model'].__class__.__name__)
        # plot_classification_learning_curve(grid_search, X, y, m['model'].__class__.__name__)
    return results





# Add the regression and classification pipelines to your main function
def run_pipeline(df):
    # Assuming you split your dataset into features (X) and target (y)
    pipeline_classification_df = create_classification_target_column(df)
    pipeline_classification_df['custom_classification_target'] = (pipeline_classification_df['custom_classification_target'] >= pipeline_classification_df['custom_classification_target'].median()).astype(int)

    c_X = pipeline_classification_df.drop(['custom_classification_target'], axis=1)
    c_y = pipeline_classification_df['custom_classification_target']

    c_X_train, c_X_test, c_y_train, c_y_test = train_test_split(c_X, c_y, test_size=0.2, random_state=42)
    # Run classification pipeline
    classification_results = run_classification_pipeline(c_X_train, c_X_test, c_y_train, c_y_test, c_X, c_y)
    classification_csv = pd.DataFrame(classification_results)
    logging.info(classification_csv)
    classification_csv.to_csv('classification_model_comparison.csv', index=False)
    logging.info("Classification Pipeline complete.")
    # Run regression pipeline
    # Assuming you split your dataset into features (X) and target (y)
    pipeline_regression_df = create_regression_target_column(df)
    pipeline_regression_df['custom_regression_target'] = (
                pipeline_classification_df['custom_regression_target'] >= pipeline_regression_df[
            'custom_regression_target'].median()).astype(int)

    r_X = pipeline_regression_df.drop(['custom_regression_target'], axis=1)
    r_y = pipeline_regression_df['custom_regression_target']

    r_X_train, r_X_test, r_y_train, r_y_test = train_test_split(r_X, r_y, test_size=0.2, random_state=42)

    regression_results = run_regression_pipeline(r_X_train, r_X_test, r_y_train, r_y_test)
    regression_csv = pd.DataFrame(regression_results)
    logging.info(regression_csv)
    regression_csv.to_csv('regression_model_comparison.csv', index=False)
    logging.info("Regression Pipeline complete.")

    plot_classification_metrics_comparison(classification_results)
    plot_regression_metrics_comparison(regression_results)
    return regression_results, classification_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    file_path = 'merged_normalized_dataset_prs.csv'
    df = pd.read_csv(file_path)
    run_pipeline(df)

