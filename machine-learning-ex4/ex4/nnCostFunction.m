function [J grad] = nnCostFunction(nn_params, ...
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
reg_param = 0;

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

% Adding bias to X
X = [ones(size(X,1),1) X];
size(X);

% Part 1: Preparing a(2), a(3) that is h(x)

a2 = sigmoid(X * Theta1');
#Adding bias to activation values
a2 = [ones(size(a2,1), 1) a2];

hypothesis = sigmoid(a2 * Theta2'); # (5000 * 10) matrix
#disp(hypothesis(1,:));

#Preparing y in 0's and 1's
y_matrix = eye(num_labels)(y,:);
##fprintf('Size of the prepared y\n');
##size(y_matrix) # (5000 * 10) matrix
##disp(y(1:5));
##disp(y_matrix(1:5,:));


#Calculating Cost

for i = 1:m
   J = J + (-1/m) * sum(y_matrix(i,:) .* log(hypothesis(i,:)) + ...
                    (1 - y_matrix(i,:)) .* log(1 - hypothesis(i,:))
             );
endfor

reg_param = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) ...
              + sum(sum(Theta2(:,2:end).^2)));

J = J + reg_param; 

% -------------------------------------------------------------
%Calculating backpropagation
% =========================================================================

for t = 1:m
  a_1 = X(t,:);  # (1 * 401) matrix
  a_1 = a_1'; # (401 * 1) vector
  z2 = Theta1 * a_1; # (25 * 1) vector
  
  a_2 = sigmoid(z2); # (25 * 1) vector
  a_2 = [1; a_2]; # (26 * 1) vector
  z3 = Theta2 * a_2; # (10 * 1) vector
  a_3 = sigmoid(z3); # (10 * 1) vector
  
  #Calculating error at output layer
  delta_3 = a_3 - y_matrix(t,:)';  # (10 * 1) vector
  #fprintf('Size of delta value: ');
  #disp(size(delta_3));
  
  #Adding bias to z2
  z2 = [1; z2]; # (26 * 1) vector
  
  #Calculating error at hidden layer.
  delta_2 = Theta2' * delta_3 .* sigmoidGradient(z2);
  # (26 * 10) * (10 * 1) .* (26 * 1)
  
  #Ignoring first term
  delta_2 = delta_2(2:end); # (25 * 1) vector
  
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  # (10 * 26) = (10 * 26) + (10 * 1) * (1 * 26)
  
  Theta1_grad = Theta1_grad + delta_2 * a_1';
  # (25 * 401) = (25 * 401) + (25 * 1) * (1 * 401) 
  
endfor

  # Unregularised gradient from accumated unregularised gradient
  Theta2_grad = (1/m) * Theta2_grad;
  Theta1_grad = (1/m) * Theta1_grad;

  #With Regularisation
  Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
  Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
  
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
