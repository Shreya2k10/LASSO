# LASSO Sparse recovery
The famous LASSO problem in ML stands for least absolute shrinkage and selection operator. It is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces.
Here I present LASSO solvers implemnted using Primal Stochastic Coordinate Descent and Proximal Gradient Descent (hyperparameter trained on train datat set provided by 3 fold cross validation). 
SubmitProxShreya and SubmitCord contain the implementation of thsse 2 methods. Respective eval files runs the submit files on train data and output a result file.
These solvers give results comparable to sklearn LASSO solver faster than it.
A detailled report on the methods is presented under the name report
