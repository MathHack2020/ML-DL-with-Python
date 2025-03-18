import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def plot_learning_curve(estimator, X, y, scoring="neg_mean_absolute_error", cv=5, 
                        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1, 
                        title="Learning Curve", figsize=(10, 6)):
    """
    Plot a learning curve for a given estimator.
    
    Parameters:
    - estimator: The model to evaluate (e.g., LinearRegression, XGBClassifier).
    - X: Training features.
    - y: Training target.
    - scoring: Scoring metric (e.g., 'neg_mean_absolute_error', 'accuracy').
    - cv: Number of cross-validation folds.
    - train_sizes: Array of training set sizes to evaluate.
    - n_jobs: Number of CPU cores to use (-1 for all).
    - title: Plot title.
    - figsize: Figure size (width, height).
    """
    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes
    )

    # Calculate mean and std (negate scores if using a 'neg_' metric)
    if scoring.startswith("neg_"):
        train_mean = -np.mean(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)
        ylabel = scoring.replace("neg_", "").replace("_", " ").title()
    else:
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        ylabel = scoring.replace("_", " ").title()

    train_std = np.std(train_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, label="Training score")
    plt.plot(train_sizes, val_mean, label="Validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel("Training Set Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()