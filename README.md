Decision Tree Classification – Tutorial & Implementation
OVERVIEW

The following repository includes two complementary resources for both understanding and applying Decision Tree classifiers:

A conceptual tutorial explaining in detail the theory behind decision trees, how they work, impurity measures, splitting criteria, pruning, and hyperparameters.

Decision Trees for Classification

A practical implementation notebook of training, evaluation, and visualization based on Python (scikit-learn) of a decision tree model using the Breast Cancer Wisconsin Diagnostic Dataset.

Notebook

Its goal is to provide both a theoretical basis and a hands-on demonstration suitable for students, educators, and beginners in machine learning.

Repository Contents
Decision Trees for Classification.pdf	Includes a full written walkthrough of the theory, impurity measures (Entropy, Gini), Information Gain, Pruning, Hyperparameters, and the teaching tips.
Decision_Tree_Tutorial.ipynb A Google Colab notebook that trains data loading, model training, accuracy evaluation, feature importance, confusion matrix visualization, and decision tree plotting.


You may find it useful to draw a number of loop traces as you watch the animations.

✔ Dataset

Uses the Breast Cancer Wisconsin Diagnostic Dataset

30 numerical features computed from digitized images of breast tissue

We are performing a binary classification: malignant versus benign.

✔ Workflow Implemented

Import necessary libraries: NumPy, Pandas, Matplotlib, scikit-learn

Loading and inspecting dataset

Splitting into training and test sets (70/30)

Training a DecisionTreeClassifier with:

criterion="gini"

max_depth=4

min_samples_leaf=5

random_state=42

Making predictions and computing accuracy (= 0.936 test accuracy)

Generating Confusion matrix

Full decision tree visualization (plot_tree)

Top-10 feature importance bar plot

Summary of the Theory 

The theoretical document lays the actual basis for decision tree construction, and it contains:

Decision Trees Overview

Recursive partitioning of feature space

Impurity reduction-based splitting

Producing interpretable if-then rules

Explanation with Examples - Titanic survival tree

Criterio de particion

Entropy & Information Gain

Gini impurity

Comparison of both metrics and associated algorithms ID3, C4.5 & CART

Hyperparameters

Explains important tuning parameters, including:

max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, criterion, ccp_alpha (cost-complexity pruning)

✂️ Pruning

Early stopping (pre-pruning)

Post-pruning (cost-complexity)

Installation & Requirements


Make sure you have Python 3.8+ and install the dependencies by running:

pip install numpy pandas matplotlib scikit-learn

To execute the notebook:

jupyter notebook Decision_Tree_Tutorial.ipynb

Overview ▶️ How to Use This Repository

Begin with the PDF tutorial to understand the concepts.

Open the notebook to follow the example hands-on.

Modify the hyperparameters to see:

Overfitting versus underfitting

Decision tree depth changes

Feature importance changes

Use this notebook as a template to apply decision trees on your own dataset.

References for notebook:
1. UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic) Data Set: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

2. scikit-learn Dataset Documentation – Breast Cancer: https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

3. Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J. (1984). Classification and Regression Trees (CART). Wadsworth Statistics/Probability Series.

4. scikit-learn Documentation – DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

5. scikit-learn User Guide – Decision Trees: https://scikit-learn.org/stable/modules/tree.html

6. scikit-learn Documentation – Confusion Matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

7. scikit-learn Tree Plotting Documentation: https://scikit-learn.org/stable/modules/tree.html#tree-visualization

8. Matplotlib Documentation: https://matplotlib.org/stable/index.html

References for Tutorial:
1. J. R. Quinlan, "Decision trees and decision-making," IEEE Transactions on Systems, Man, and Cybernetics, vol. 20, no. 2, pp. 339–346, 1990.
2. S. R. Safavian and D. Landgrebe, "A survey of decision tree classifier methodology," IEEE Transactions on Systems, Man, and Cybernetics, vol. 21, no. 3, pp. 660–674, 1991.
3. S. B. Gelfand, C. S. Ravishankar and E. J. Delp, "An iterative growing and pruning algorithm for classification tree design," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 2, pp. 163–174, 1991.
4. F. Esposito, D. Malerba and G. Semeraro, "A comparative analysis of methods for pruning decision trees," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 19, no. 5, pp. 476–491, 1997.
5. B. Kim and D. A. Landgrebe, "Hierarchical classifier design in high-dimensional numerous class cases," IEEE Transactions on Geoscience and Remote Sensing, vol. 29, no. 4, pp. 518–528, 1991.
6. T. K. Ho, "The random subspace method for constructing decision forests," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 20, no. 8, pp. 832–844, 1998.
7. L. Rokach and O. Maimon, "Top-down induction of decision trees classifiers — a survey," IEEE Transactions on Systems, Man, and Cybernetics, Part C: Applications and Reviews, vol. 35, no. 4, pp. 476–487, 2005.
