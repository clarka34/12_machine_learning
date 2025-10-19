from sklearn.model_selection import learning_curve, LearningCurveDisplay

def plot_lc(estimator, X, y, title=None, axes=None, ylim=None, cv=None,
            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), random_state=None):
    # create your method to plot your learning curves using the sklearn method learning_curve()
    LearningCurveDisplay.from_estimator(estimator, X, y, train_sizes=train_sizes, cv=cv, 
                                        n_jobs=n_jobs, random_state=random_state, ax=axes, 
                                        std_display_style='fill_between')
    if axes is not None:
        handles, label = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["Training Score", "Test Score"])
    if (title is None) and (axes is None):
        plt.title(f"Learning Curve for {estimator.named_steps['classifier']}")
    elif (title is None) and (axes is not None):
        ax.set_title(f"Learning Curve for {estimator.named_steps['classifier']}")
    elif (title is not None) and (axes is None):
        plt.title(title)
    else:
        ax.set_title(title)


from sklearn.model_selection import learning_curve

def plot_lc(estimator, X, y, title=None, axes=None, ylim=None, cv=None,
            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # create your method to plot your learning curves using the sklearn method learning_curve()
    learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs)
    if axes is not None:
        handles, label = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["Training Score", "Test Score"])
    if (title is None) and (axes is None):
        plt.title(f"Learning Curve for {estimator.named_steps['classifier']}")
    elif (title is None) and (axes is not None):
        ax.set_title(f"Learning Curve for {estimator.named_steps['classifier']}")
    elif (title is not None) and (axes is None):
        plt.title(title)
    else:
        ax.set_title(title)


def plot_learning_curve2(estimator, x, y, training_set_sizes, cv=None):
    
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

