function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape( nn_params(1:hidden_layer_size * (input_layer_size + 1)) , ...
                 hidden_layer_size, (input_layer_size + 1) );

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Y = zeros(m,num_labels);

for i=1:m
    j = y(i);
    Y(i,j) = 1;
end;

one_col = ones(m,1);
X = [one_col, X];
hidden_activations = sigmoid(X * Theta1');

hidden_activations = [one_col,hidden_activations];

h = sigmoid(hidden_activations*Theta2');

for p=1:m
    for q=1:num_labels
        J = J + ((Y(p,q))*log(h(p,q)) + ((1-Y(p,q)))*log(1 - h(p,q)));
    end;
end;
J = -1 * J/m;

%regularized cost implementation
temp = Theta1.^2;
temp(:,1) = zeros(hidden_layer_size,1);
reg_cost = sum(sum(temp));
temp2 = Theta2.^2;
temp2(:,1) = zeros(num_labels,1);
reg_cost = reg_cost + sum(sum(temp2));
J = J + (lambda*reg_cost/(2*m));

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for p1=1:m
    a1 = X(p1,:);
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [1,a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    
    delta3 = a3 - Y(p1,:);
    %fprintf('Lsize of a1 is :\n')
    %display(size(a1));
    %fprintf('Lsize of D1 is :\n')
    %display(size(D1));
    %fprintf('Lsize of theta 1 is :\n')
    %display(size(Theta1));
    %display(size(z2));
    delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
    D1 = D1 + delta2' * a1;
    D2 = D2 + delta3' * a2;
end;
%fprintf('Lsize of zeros is :\n')
%display(size(zeros(size(Theta1,2),1)));
%fprintf('Lsize of t1 is :\n')
%display(size(Theta1(:,2:end)));

temp1 = Theta1 ./ m;
temp2 = Theta2 ./ m;
temp1(:,1) = zeros(size(Theta1,1),1);
temp2(:,1) = zeros(size(Theta2,1),1);
%display(temp1);

Theta1_grad = D1 / m + lambda*temp1;
Theta2_grad = D2 / m + lambda*temp2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
