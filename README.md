# CS4220 Machine Learning 1 Lab Exercises

### *Delft University of Technology 23/24*


## 1. Basics of Machine Learning

- **Optimal decision boundary** (where posteriors `P(y1|x) == P(y2|x)`) corresponds to **Bayes error** (i.e. minimal error)
-  **Quadratic Classifier (QDA)** has a quadratic decision boundary, due to limited data. 
    - In the limit, the solutions by LDA and QDA will coincide
- Hyperparameters (such as `h` in Parzen estimator) should be tuned on test sets independent from training data
    - Optimal value can be found by plotting values (x-axis) against the log-likelihood (y-axis) using elbow method
- k-NN and NMC are sensitive to **scaling** of features, because these methods do not estimate covariance matrices


## 2. Linear Regression & Linear Classifiers


## 3. Losses, Regularization, Evaluation


## 4. Probabilistic Models & Clustering


## 5. Complexity


## 6. Reducing & Combining