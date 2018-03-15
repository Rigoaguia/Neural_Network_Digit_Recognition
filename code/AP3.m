% Universidade Federal de Mato Grosso
% Instituto de Engenharia
% Inteligência Artificial - 2017/1
%
% Script Octave/MATLAB que deve ser utilizado para testar a implementação 
% do exercício.
% Este arquivo não deve ser  alterado.
%
% Você deve alterar apenas os arquivos: 
% 
%     predict.m
%     cost_function.m
%

clear ; close all; clc

%% Parametros da rede
input_layer_size  = 400;  % imagens de 20x20 
hidden_layer_size = 25;   % 25 unidades na camada intermediária
num_labels = 10;          % 10 classe, de 1 a 10
                          % (note que o digito "0" foi mapeado para a classe 10)

%% =========== Carregando e Visualisando os Dados  =============


fprintf('Carregando e Visualisando os Dados ...\n')

load('ex5data.mat');
m = size(X, 1);

% Seleciona 100 imagens aleatoriamente
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Programa parado. Digite enter para continuar.\n');
pause;


%% ================ Carregando Parâmetros ================


fprintf('\nCarregando parâmetros salvos ...\n')

load('ap3weights.mat');

nn_params = [Theta1(:) ; Theta2(:)];


%% ============= Parte 1: Testando a implementação da classificação =============
%  
%  

pred = predict(Theta1, Theta2, X);

fprintf('\nAcuracia do treinamento: %f\n', mean(double(pred == y)) * 100);

fprintf('Programa parado. Digite enter para continuar.\n');
pause;


% Veja  10 exemplos aleatorios

%  Randomly permute examples
rp = randperm(m);

for i = 1:10
    % Display
    fprintf('\nMostrando Imagem\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nEsta imagem foi classificada como: %d (digito %d)\n', pred, mod(pred, 10));

    % Pause
	fprintf('Programa parado. Digite enter para continuar.\n');
    pause;
end



%% ======= Parte 2a: Testando implementação da função custo (sem regularizacao) =======
%  
%
fprintf('\nFunção de custo (sem regularização)...\n')

% desconsidera a regularização.
lambda = 0;

J = cost_function(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Custo calculado considerado os valores salvos em ex5weights: %f '...
         '\n(O valor correto deve ser próximo de 0.287629)\n'], J);

fprintf('Programa parado. Digite enter para continuar.\n');
pause;

%% =============== Parte 2b: Testando implementação da função custo (com regularizacao) ===============
%  

fprintf('\nFunção de custo (com regularização)...\n')

lambda = 1;

J = cost_function(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Custo calculado considerado os valores salvos em ex5weights: %f '...
         '\n(O valor correto deve ser próximo de 0.383770)\n'], J);

fprintf('Programa parado. Digite enter para continuar.\n');
pause;




%% =============== Parte 2c: Testando implementação do Backpropagation (sem regularização)===============
%  
%
fprintf('\nTestando Backpropagation (sem Regularização) ... \n')

lambda = 0;
checkNNGradients;

fprintf('Programa parado. Digite enter para continuar.\n');
pause;


%% =============== Parte 2d: Testando implementação do Backpropagation (com regularização)= ===============
%  
%

fprintf('\nTestando Backpropagation (com Regularização) ... \n')

lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = cost_function(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCusto: %f ' ...
         '\n(O valor correto deve ser próximo de 0.576051)\n\n'], debug_J);



