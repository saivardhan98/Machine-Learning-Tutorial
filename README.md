**Decision Tree Classification – Tutorial & Implementation**
**OVERVIEW**

The following repository includes two complementary resources for both understanding and applying Decision Tree classifiers:

A conceptual tutorial explaining in detail the theory behind decision trees, how they work, impurity measures, splitting criteria, pruning, and hyperparameters.

Decision Trees for Classification

A practical implementation notebook of training, evaluation, and visualization based on Python (scikit-learn) of a decision tree model using the Breast Cancer Wisconsin Diagnostic Dataset.

Its goal is to provide both a theoretical basis and a hands-on demonstration suitable for students, educators, and beginners in machine learning.

**Repository Contents**
Decision Trees for Classification.pdf	Includes a full written walkthrough of the theory, impurity measures (Entropy, Gini), Information Gain, Pruning, Hyperparameters, and the teaching tips.
Decision_Tree_Tutorial.ipynb A Google Colab notebook that trains data loading, model training, accuracy evaluation, feature importance, confusion matrix visualization, and decision tree plotting.


You may find it useful to draw a number of loop traces as you watch the animations.

**Dataset**

Uses the Breast Cancer Wisconsin Diagnostic Dataset

30 numerical features computed from digitized images of breast tissue

We are performing a binary classification: malignant versus benign.

**Workflow Implemented**

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

**Pruning**

Early stopping (pre-pruning)

Post-pruning (cost-complexity)

**Installation & Requirements**


Make sure you have Python 3.8+ and install the dependencies by running:

pip install numpy pandas matplotlib scikit-learn

**To execute the notebook:**

jupyter notebook Decision_Tree_Tutorial.ipynb

Overview How to Use This Repository

Begin with the PDF tutorial to understand the concepts.

Open the notebook to follow the example hands-on.

**Modify the hyperparameters to see:**

Overfitting versus underfitting

Decision tree depth changes

Feature importance changes

Use this notebook as a template to apply decision trees on your own dataset.
