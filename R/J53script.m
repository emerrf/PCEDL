cd C:\Users\emerrf\Documents\GitHub\PCEDL_local\R
clc; clear all;
rng(1)

% Load data
windpw = csvread("kernplus_windpw.csv", 1);

X_train = windpw(1:900, 1:5);
Y_train = windpw(1:900, 6);
X_test = windpw(901:1000, 1:5);
Y_test = windpw(901:1000, 6);

% [trainInd,valInd,testInd] = divideind(1000,1:800,801:900,901:1000);
% X_train = windpw(trainInd, 1:5);
% X_val = windpw(valInd, 1:5);
% X_test = windpw(testInd, 1:5);
% Y = windpw(:, 6);

% Normalise data
X_train = (X_train - mean(X_train))./std(X_train);
% X_val = (X_val - mean(X_val))./std(X_val);
X_test = (X_test - mean(X_test))./std(X_test);
% X = [X_train; X_val; X_test];


X = X_train;
Y = Y_train;

% Create a Fitting Network
hiddenLayerSize = 32;
trainFcn = 'trainlm'; %  Levenberg-Marquardt backpropagation.
net = fitnet(hiddenLayerSize,trainFcn);

% net.divideFcn = 'divideind';
% net.divideParam.trainInd = trainInd;
% net.divideParam.valInd   = valInd;
% net.divideParam.testInd  = testInd;

% Train the Network
[net,tr] = train(net,X',Y');

mse = mean(power(Y_test - sim(net, X_test')',2));
rmse = sqrt(mse);

% mse =
%   163.1141
% rmse =
%    12.7716
