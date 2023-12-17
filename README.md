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

- Differently from labelled datasets (often numeric labels `y`) in supervised classification tasks, **regression datasets** contain inputs `x` (uniformly distributed e.g.) with a corresponding `y` depdendent on `x` (`y = x^2 + eps` where `eps` can be some Gaussian noise e.g.)
- We can fit linear functions under the **squared loss** using `w = (X^TX)^-1 * X^T * Y`
    - Typically we have an intercept as well giving `w^T = [w0 w1]` which can be plugged into `f(x) = w1 * x + w0` 
    - Note we need to add a bias term to X
- **Polynomial Regression** fits a polynomial of some *maximal* degree to the data in a least squares sense
    - *Note: even though this results in a non-linear function in x, the regression function is linear in the unknown parameters w estimated from the data.*
    - If there are 4 data points, one needs at least a third-order polynomial to fit these (whether there is a bias term does not matter)
    - Note that linear regressors (`linearr`) do not take into cross-terms, leading to high error rates when trained on (non-linear data) that contain multiplications of `x_i` terms: e.g. data modelled as `y = sin(x1)sin(x2)`
- **Decision boundary** of **linear classifier** is found at `f(x) = y = 0`
    - `y = ax + b` with intercept `b = -w0 / w2` and slope `a = -(w0/w2)/(w0/w1)` 


## 3. Losses, Regularization, Evaluation

- Regularization aims to stabilize objective functions to prevent overfitting - possibly caused by curse of dimensionality or lack of data (multicollinearity)
    - **Ridge regularization (L2)** keeps weights small
    - **Lasso regularization (L1)** introduces sparsity and can get rid of redundant features
    - Whereas $w_i$ entries can become 0 for large $\lambda$ (i.e. small $\tau$) in L1 regularization, this is impossible for L2
- The optimal regularization parameter $\lambda$ can be found using **cross-validation** (hyperparameter tuning)
    - **K-fold Cross Validation**: N chunks used for training independent classifiers (take avg of `N` error estimates), 1 chunk used for evaluating (compared against *true error* on full training set)
    - **Leave-one-out-Procedure**: Single test object
- **Learning curves** show classification errors against the training set size
    - The discrepancy between *true error* and *apparent error* is **overfitting**
- **Feature curves** show how the classification error varies with varying numbers of feature dimensionality
    - **Curse of dimensionality** = error goes up after a certain dimensionality threshold 
- *Variabililty in error* is larger for small training sets
    - The more complex the model, the more training samples needed! 
    - *Large, independent test sets* yield an unbiased and small variance error estimate (but worse classifier due to less data used for training..)
    - *Small, independent test sets* yield an unbiased and large variance error estimate






## 4. Probabilistic Models & Clustering


## 5. Complexity


## 6. Reducing & Combining