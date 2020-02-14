function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_train = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_train = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
col = size(sigma_train, 2);
row = size(C_train, 2);
prediction_error = zeros(row, col);

for i = 1:row
  for j = 1:col
    model_train = svmTrain(X, y, C_train(i), ...
    @(x1, x2) gaussianKernel(x1, x2, sigma_train(j)));
    predict_crossval = svmPredict(model_train, Xval);
    ##fprintf('Model prediction^^~~');
    ##disp(predict_crossval);
    prediction_error(i, j) = mean(double(predict_crossval ~= yval));
    #disp(prediction_error);
  endfor
endfor
disp(prediction_error);

[predErrorMin, row_id] = min(min(prediction_error,[], 2));
[predErrorMin, col_id] = min(min(prediction_error,[], 1));

#fprintf('Row id of the lowest value: %i',row_id);
#fprintf('Column id of the lowest value: %i',col_id);

fprintf('C value with lowest prediction error- %f\n', C_train(row_id));
fprintf('Sigma value with lowest prediction error- %f\n', sigma_train(col_id));

C = C_train(row_id);
sigma = sigma_train(col_id);



% =========================================================================

end
