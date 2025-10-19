# Plot learning curves
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

def plot_learning_curve(estimator, x, y, training_set_sizes, cv=None):
    
    # Shuffle the dataset before looping
    x, y = shuffle(x, y)

    train_scores_means = []
    val_scores_means = []

    for s in training_set_sizes:
        x_subset = x[:s]
        y_subset = y[:s]

        # This is what happens inside cross_val_score:

        train_scores = []
        val_scores = []

        # Define the K-Folds cross-validator
        kf = StratifiedKFold(n_splits=cv, shuffle=True)

        for train_index, val_index in kf.split(x_subset, y_subset):
            # Split the dataset in the i-th fold
            x_train, x_val = x_subset[train_index], x_subset[val_index]
            y_train, y_val = y_subset[train_index], y_subset[val_index]

            # Fit the model on the training set of the kth fold
            estimator.fit(x_train, y_train)

            # Evaluate and store the result
            train_scores += [metrics.accuracy_score(y_train, estimator.predict(x_train))]
            val_scores += [metrics.accuracy_score(y_val, estimator.predict(x_val))]
        
        # Cross val is finished, store the means
        train_scores_means += [np.mean(train_scores)]
        val_scores_means += [np.mean(val_scores)]

    # Plot the mean accuracy on the cv folds
    plt.plot(training_set_sizes, train_scores_means, 'o-', color="r", label="Training score")
    plt.plot(training_set_sizes, val_scores_means, 'o-', color="g", label="Cross-validation score")
    plt.legend()

    # Plot the mean accuracy on the cv folds and +- one std
    #plt.fill_between(training_set_sizes, np.mean(train_scores) - np.std(train_scores), np.mean(train_scores) + np.std(train_scores), alpha=0.1, color="r")
    #plt.fill_between(training_set_sizes, np.mean(val_scores) - np.std(val_scores), np.mean(val_scores) + np.std(val_scores), alpha=0.1, color="g")
