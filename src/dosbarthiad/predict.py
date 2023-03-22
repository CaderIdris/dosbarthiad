from typing import Any

import pandas as pd
import sklearn.discriminant_analysis as da
import sklearn.ensemble as en
import sklearn.linear_model as lm
import sklearn.metrics as met
import sklearn.neighbors as nb
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.tree as tr


class Predict:
    """
    Classifies an object based on a group of features with multiple
    scikit-learn classifiers.
    Effectively a very high level scikit-learn wrapper.

    ```

    Attributes
    ----------
    __train : dict[int, pd.DataFrame]
        Dict of dataframes containing training data, int keys represent
        a different fold of k-folds validation
    __test : dict[int, pd.DataFrame]
        Dict of dataframes containing testing data, int keys represent
        a different fold of k-folds validation
    __y_col : dict[int, pd.Series]
        Actual classifications of training data used to train classifiers.
        Int keys represent a different fold of k-folds validation.
    __test_actual : dict[int, pd.Series]
        Actual classifications of testing data used to validate predictions.
        Int keys represent a different fold of k-folds validation.
    __predicted_results : dict[int, pd.DataFrame]
        Predicted classifications for testing data, each column represents
        a different technique. Int keys represent a different fold of k-folds
        validation.
    __classifiers : dict[int, dict[str, Any scikit-learn classifier]]
        Trained classifier objects.Int keys represent a different fold of
        k-folds validation.
    __scores : dict[int, pd.DataFrame]
        Scores of predicted vs actual classifications. Each column is a
        different scoring method, each row is a different classification method

    Methods
    -------
    _skl_class(name, classifier)
        Meta function that accepts any scikit-learn method that predicts the
        class of an object based on a group of features
    log_reg(name="Logistic Regression", max_iter=1000, random_state=62, **kw)
        Performs logistic regression using scikit-learns LogisticRegression
        class
    log_reg_cv(
        name="Logistic Regression CV", max_iter=1000, random_state=62, **kw
        )
        Performs cross-validated logistic regression using scikit-learns
        LogisticRegressionCV class
    ridge(name="Ridge Regression", random_state=62, **kw)
        Performs ridge classification using scikit-learns RidgeClassifier class
    ridge_cv(name="Ridge Regression CV", **kw)
        Performs ridge classification using scikit-learns RidgeClassifierCV
        class
    passive_aggressive(name="Passive Aggressive", random_state=62, **kw)
        Perform passive aggressive classification using scikit-learns
        PassiveAggressiveClassifier class
    perceptron(name="Perceptron", random_state=62, **kw)
        Uses a perceptron to classify data using scikit-learns Perceptron class
    sgd(name="Stochastic Gradient Descent", random_state=62, **kw)
        Performs stochastic gradient descent to classify data using
        scikit-learns SGDClassifier class
    knn(name="k-Nearest Neighbours', weights='distance', **kw)
        Performs k-Nearest Neighbours classification using scikit-learns
        KNeighborsClassifier class
    decision_tree(name="Decision Tree", random_state=62, **kw)
        Performs classification with a decision tree using scikit-learns
        DecisionTreeClassifier class
    extra_tree(name="Extra Tree", random_state=62, **kw)
        Performs classification with an extra tree using scikit-learns
        ExtraTreeClassifier class
    random_forest(name="Random Forest", random_state=62, **kw)
        Performs random forest classification using scikit-learns
        RandomForestClassifier class
    extra_tree_ensemble(name="Extra Trees (Ensemble)", random_state=62, **kw)
        Performs classification using an ensemble of extra trees using
        scikit-learns ExtraTreesClassifier class
    gradient_boost(name="Gradient Boost", random_state=62, **kw)
        Performs gradient boost classification using scikit-learns
        GradientBoostingClassifier class
    hist_gradient_boost(
        name="Histogram-based Gradient Boosting, random_state=62, **kw
        )
        Performs histogram based gradient boost classification using
        scikit-learns HistGradientBoostingClassifier class
    mlp(
        name="Multi-layer Perceptron", solver="adam", max_iter=2500,
        random_state=62, **kw
        )
        Uses a multi-layer perceptron to classify data using scikit-learns
        MLPClassifier
    svc(
        name="Support Vector Classification", max_iter=5000, random_state=62,
        **kw)
        Performs support vector classification using scikit-learns LinearSVC
        class
    linear_svc(
        name="Linear Support Vector Classification", max_iter=5000,
        random_state=62, dual=False, **kw
        )
        Performs linear support vector classification using scikit-learns
        LinearSVC class
    nu_svc(name="Nu-Support Vector Classification", random_state=62, **kw
        )
        Performs nu-support vector classification using scikit-learns
        NuSVC class
    lda(name="Linear Discriminant Analysis", **kw)
        Uses linear discriminant analysis to classify data using scikit-learns
        LinearDiscriminantAnalysis class
    ada_boost(classifier, name, random_state=62, **kw)
        Uses Ada boosting in tandem with provided classifier using
        scikit-learns AdaBoostClassifier class
    bagging(classifier, name, random_state=62, **kw)
        Uses bagging in tandem with provided classifier using
        scikit-learns BaggingClassifier class
    __calculate_scores()
        Uses multiple metrics to assess predicted results
    __set_pred(pred)
        Sets predicted values
    return_pred()
        Returns predicted values
    return_actual()
        Return actual values
    return_scores()
        Returns assessment of predicted results using multiple metrics
    return_classifiers()
        Returns the trained classifiers
    """
    def __init__(
            self,
            train: dict[int, pd.DataFrame],
            test: dict[int, pd.DataFrame],
            y_col: str
            ):
        """
        Constructs the predict object

        Parameters
        ----------
        train : dict[int, pd.DataFrame]
            Training data. Int keys represent a different fold of k-folds
            validation
        test : dict[int, pd.DataFrame]
            Testing data. Int keys represent a different fold of k-folds
            validation
        y_col : str
            Name of the column with the actual classifications of the data
        """
        self.__train: dict[int, pd.DataFrame] = dict()
        self.__test: dict[int, pd.DataFrame] = dict()

        self.__y_col: dict[int, pd.Series] = dict()
        self.__test_actual: dict[int, pd.Series] = dict()

        self.__predicted_results: dict[int, pd.DataFrame] = dict()
        self.__classifiers: dict[int, dict[str, Any]] = dict()

        self.__scores: dict[int, pd.DataFrame] = dict()

        for i in train.keys():
            self.__train[i] = train[i].drop([y_col], axis=1)
            self.__test[i] = test[i].drop([y_col], axis=1)
            self.__y_col[i] = train[i].loc[:, y_col]
            self.__test_actual[i] = test[i].loc[:, y_col]
            self.__predicted_results[i] = pd.DataFrame(
                index=self.__test_actual[i].index
            )
            self.__scores[i] = pd.DataFrame()
            self.__classifiers[i] = dict()

    def _skl_class(
            self,
            name: str,
            classifier: Any
            ):
        """
        Meta function that accepts any scikit-learn method that predicts the
        class of an object based on a group of features

        Parameters
        ----------
        name : str
            Name of the classification technique that the results are saved
            under
        classifier : Any scikit-learn classification method
        """
        for i in self.__train.keys():
            classifier.fit(
                    self.__train[i],
                    self.__y_col[i]
                    )

            self.__predicted_results[i][name] = pd.Series(
                    data=classifier.predict(self.__test[i]),
                    index=self.__test[i].index
                    )
            self.__classifiers[i][name] = classifier

    def log_reg(
            self,
            name: str = "Logistic Regression",
            max_iter=1000,
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs logistic regression using scikit-learns LogisticRegression
        class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Logistic Regression"
        max_iter : int, optional
            Maximum number of iterations before completing
            Default is 1000
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                lm.LogisticRegression(
                    random_state=random_state,
                    max_iter=max_iter,
                    **kwargs
                    )
                )

    def log_reg_cv(
            self,
            name: str = "Logistic Regression CV",
            max_iter=1000,
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs cross-validated logistic regression using scikit-learns
        LogisticRegressionCV class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Logistic Regression CV"
        max_iter : int, optional
            Maximum number of iterations before completing
            Default is 1000
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                lm.LogisticRegressionCV(
                    random_state=random_state,
                    max_iter=max_iter,
                    **kwargs
                    )
                )

    def ridge(
            self,
            name: str = "Ridge Regression",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs ridge regression using scikit-learns
        RidgeClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Ridge Regression"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                lm.RidgeClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def ridge_cv(
            self,
            name: str = "Ridge Regression CV",
            **kwargs
            ):
        """
        Performs cross-validated ridge regression using scikit-learns
        RidgeClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Ridge Regression CV"
        """
        self._skl_class(
                name,
                lm.RidgeClassifierCV(**kwargs)
                )

    def passive_aggressive(
            self,
            name: str = "Passive Aggressive",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs passive aggressive classification using scikit-learns
        PassiveAggressiveClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Passive Aggressive"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                lm.PassiveAggressiveClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def perceptron(
            self,
            name: str = "Perceptron",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        Perceptron class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Perceptron"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                lm.Perceptron(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def sgd(
            self,
            name: str = "Stochastic Gradient Descent",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs stochastic gradient descent classification using scikit-learns
        SGDClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Stochastic Gradient Descent"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                lm.SGDClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def knn(
            self,
            name: str = "k-Nearest Neighbors",
            weights: str = 'distance',
            **kwargs
            ):
        """
        Performs k-nearest neighbours classification using scikit-learns
        KNeighborsClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "k-Nearest Neighbors"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                nb.KNeighborsClassifier(
                    weights=weights,
                    **kwargs
                    )
                )

    def decision_tree(
            self,
            name: str = "Decision Tree",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        DecisionTreeClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Decision Tree"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                tr.DecisionTreeClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def extra_tree(
            self,
            name: str = "Extra Tree",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        ExtraTreeClassifier class

        Recommended only to be used in an ensemble.

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Extra Tree"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                tr.ExtraTreeClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def random_forest(
            self,
            name: str = "Random Forest",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        RandomForestClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Random Forest"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                en.RandomForestClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def extra_tree_ensemble(
            self,
            name: str = "Extra Trees (Ensemble)",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        ExtraTreesClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Extra Trees (Ensemble)"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                en.ExtraTreesClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def gradient_boost(
            self,
            name: str = "Gradient Boost",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        GradientBoostingClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Gradient Boost"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                en.GradientBoostingClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def hist_gradient_boost(
            self,
            name: str = "Histogram-based Gradient Boosting",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        HistGradientBoostingClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Histogram-based Gradient Boosting"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                en.HistGradientBoostingClassifier(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def mlp(
            self,
            name: str = "Multi-layer Perceptron",
            solver: str = 'adam',
            max_iter: int = 2500,
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        MLPClassifier class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Multi-layer Perceptron"
        solver : str, optional
            Solver to define for MLPClassifier
            Default is 'adam'
        max_iter : int, optional
            Maximum number of iterations before completing
            Default is 2500
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                nn.MLPClassifier(
                    random_state=random_state,
                    solver=solver,
                    max_iter=max_iter,
                    **kwargs
                    )
                )

    def svc(
            self,
            name: str = "Support Vector Classification",
            max_iter: int = 5000,
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        SVC class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Support Vector Classification"
        max_iter : int, optional
            Maximum number of iterations before completing
            Default is 2500
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                svm.SVC(
                    random_state=random_state,
                    max_iter=max_iter,
                    **kwargs
                    )
                )

    def linear_svc(
            self,
            name: str = "Linear Support Vector Classification",
            max_iter: int = 5000,
            random_state: int = 62,
            dual: bool = False,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        LinearSVC class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Linear Support Vector Classification"
        max_iter : int, optional
            Maximum number of iterations before completing
            Default is 2500
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                svm.LinearSVC(
                    random_state=random_state,
                    dual=dual,
                    max_iter=max_iter,
                    **kwargs
                    )
                )

    def nu_svc(
            self,
            name: str = "Nu-Support Vector Classification",
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        NuSVC class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Nu-Support Vector Classification"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                svm.NuSVC(
                    random_state=random_state,
                    **kwargs
                    )
                )

    def lda(
            self,
            name: str = "Linear Discriminant Analysis",
            **kwargs
            ):
        """
        Performs classification using scikit-learns
        LinearDiscriminantAnalysis class

        Parameters
        ----------
        name : str, optional
            Name to use for the results
            Default is "Nu-Support Vector Classification"
        """
        self._skl_class(
                name,
                da.LinearDiscriminantAnalysis(
                    **kwargs
                    )
                )

    def ada_boost(
            self,
            classifier: Any,
            name: str,
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs ada boosting on provided classification method using
        scikit-learns AdaBoostClassifier class

        Works best on weak models (i.e. shallow decision trees)

        Parameters
        ----------
        classifier : Any scikit-learn classifier
            The classifier to be boosted
        name : str, optional
            Name to use for the results
            Default is "Nu-Support Vector Classification"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                en.AdaBoostClassifier(
                    classifier,
                    random_state=random_state,
                    **kwargs
                    )
                )

    def bagging(
            self,
            classifier: Any,
            name: str,
            random_state: int = 62,
            **kwargs
            ):
        """
        Performs bagging on provided classification method using
        scikit-learns BaggingClassifier class to reduce variance of
        base estimator

        Works best on strong models (i.e. fully developed decision trees)

        Parameters
        ----------
        classifier : Any scikit-learn classifier
            The classifier to be boosted
        name : str, optional
            Name to use for the results
            Default is "Nu-Support Vector Classification"
        random_state : int, optional
            Seed for random number generator
            Default is 62
        """
        self._skl_class(
                name,
                en.BaggingClassifier(
                    classifier,
                    random_state=random_state,
                    **kwargs
                    )
                )

    def __calculate_scores(self):
        """
        Use multiple metrics to determine the performance of
        predicted vs actual classifications
        """
        for i in self.__train.keys():
            actual = self.__test_actual[i]
            tests: dict[str, Any] = {
                'Accuracy Score': met.accuracy_score,
                'Average Precision Score': met.average_precision_score,
                'Brier Score Loss': met.brier_score_loss,
                'F1 Score': met.f1_score,
                'Hamming Loss': met.hamming_loss,
                'Log Loss': met.log_loss,
                'Precision': met.precision_score,
                'Recall': met.recall_score,
                'Zero-one Loss': met.zero_one_loss
                    }
            for name, pred in self.__predicted_results[i].items():
                comparison = (actual, pred)
                for test, func in tests.items():
                    self.__scores[i].loc[name, test] = func(*comparison)
                confusion_matrix = met.confusion_matrix(*comparison)
                self.__scores[i].loc[
                        name,
                        'Confusion Matrix (True Negatives)'
                        ] = confusion_matrix[0][0]
                self.__scores[i].loc[
                        name,
                        'Confusion Matrix (False Negatives)'
                        ] = confusion_matrix[1][0]
                self.__scores[i].loc[
                        name,
                        'Confusion Matrix (False Positives)'
                        ] = confusion_matrix[0][1]
                self.__scores[i].loc[
                        name,
                        'Confusion Matrix (True Positives)'
                        ] = confusion_matrix[1][1]

    def set_pred(self, pred: dict[int, pd.DataFrame]):
        """
        Set predicted classifications

        Parameters
        ----------
        pred : dict[int, pd.DataFrame]
            Pre-generated predicted classifications
        """
        self.__predicted_results = pred

    def return_pred(self) -> dict[int, pd.DataFrame]:
        """
        Returns predicted classifications
        """
        return self.__predicted_results

    def return_actual(self) -> dict[int, pd.DataFrame]:
        """
        Returns actual classifications
        """
        return self.__test_actual

    def return_scores(self) -> dict[int, pd.DataFrame]:
        """
        Calculates scores and returns them
        """
        self.__calculate_scores()
        return self.__scores

    def return_classifiers(self) -> dict[int, dict[str, Any]]:
        """
        Returns the classifier objects
        """
        return self.__classifiers
