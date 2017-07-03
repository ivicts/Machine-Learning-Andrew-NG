function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
htheta = zeros(m,1);
x = X*theta;
sumtheta = zeros(length(theta)-1,1);

for i = 1:m
   htheta(i) = sigmoid(x(i));
end

for i= 2:length(theta)
  sumtheta(i) = theta(i)*theta(i);
end

reg = sum(sumtheta);

J = sum(-y.*log(htheta)-(1-y).*log(1-htheta))/m + (lambda/(2*m))*reg;

%J = sum(-y.*log((sigmoid(X*theta))')-(1-y).*log(1-(sigmoid(X*theta))'))/m;

grad = zeros(size(theta));

grad(1) = sum((htheta-y).*X(:,1))/m;

for j = 2:length(theta)
        grad(j) = sum((htheta-y).*X(:,j))/m +lambda*theta(j)/m;
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
