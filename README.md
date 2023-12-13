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


## 4. Probabilistic Models & Clustering


## 5. Complexity


## 6. Reducing & Combining