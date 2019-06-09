function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


J = (-y' * log(sigmoid(X * theta)) - (1 - y)' * log(1 - sigmoid(X * theta))) / m + lambda / (2*m) * theta' * theta - lambda / (2*m) * theta(1, 1)^2;

grad = 1/m * X' * (sigmoid(X * theta) - y) + lambda / m * theta;

grad(1, 1) = grad(1,1) - lambda / m * theta(1, 1);


end
