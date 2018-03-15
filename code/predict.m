function [p,y3,z2,entrada,y2] = predict(Theta1, Theta2, X)
% função feedforward
% theta1 = 25x401, theta2 = 10x26 e X = 5000x400
%displayData(displayData(X(1,:)));
%clc
%load('ap3weights.mat','Theta1','Theta2');
%load('ex5data.mat','X');
% bias para camada de entrada
b1 = repmat(1,[size(X,1) 1])';
% acrescentando a bias e a entrada 'X' na camada de entrada
entrada = [b1;X'];
size(entrada);
% ponderado as entrada com seus respectivos pesos Theta1 na camada oculta
z2 = entrada' * Theta1';
y =  (sigmoid(z2))'; 
% size(y);
% size(z2);
%z2 = 5000x25
% bias para camada de saída
b2 = repmat(1,[size(y,2) 1])';
% acrescentando a bias e a saída na camada de saída
y2 = [b2;y];

size(y2);
% saída da comada de saída
y3 = sigmoid(Theta2*y2);
size(y3);
% y3 = 10x5000
[v,p] = max(y3',[],2);



end