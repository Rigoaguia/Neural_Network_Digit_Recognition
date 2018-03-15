% Universidade Federal de Mato Grosso
% Instituto de Engenharia
% Inteligência Artificial - 2016/2
%
% Script Octave/MATLAB que deve ser utilizado para treinar a rede neural
%
% Você DEVE alterar os valores da variável lambda (linha 40)
% 
%  
%

clear ; close all; clc

%% Parametros da rede
input_layer_size  = 400;  % imagens de 20x20 
hidden_layer_size = 25;   % 25 unidades na camada intermediária
num_labels = 10;          % 10 classe, de 1 a 10
                          % (note que o digito "0" foi mapeado para a classe 10)

%% =========== Carregando os Dados  =============


fprintf('Carregando os Dados ...\n')

load('ex5data.mat');
m = size(X, 1);




%% =================== Treinando a rede neural ===================
%  
%
fprintf('\nTreinando a rede neural... \n')


options = optimset('MaxIter', 50);

%  should also try different values of lambda
lambda = 1;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% 
costFunction = @(p) cost_function(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Função de otimização
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));






