function [J grad] = cost_function(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%COST_FUNCTION implementa a função de custo da rede neural
%   [J grad] = COST_FUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula a função de custo  e o gradiente da rede. 
%

% Não altere!!
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Variáveis úteis
m = size(X, 1);

% Você deve preencher as seguintes variáveis corretamente
% J = 0;
% Theta1_grad = zeros(size(Theta1)); % gradiente de Theta1
% Theta2_grad = zeros(size(Theta2)); % gradiente de Theta2



% Mudança de representação de y para um vetor Y
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% =============== Sua implementação deve ser vir aqui ==================

% funcao custo
% theta1 = 25x401, theta2 = 10x26
[~,y3,~,~,~] = predict(Theta1,Theta2,X);

y3 = y3';

J = sum(sum(-(Y.*log(y3))-(1-Y).*log(1-y3)))/m;
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


% backpropagation

[~,y3,z2,entrada,y2] = predict(Theta1,Theta2,X);

E3 = (y3 - Y');
ax = (Theta2')*E3;
ax = ax(2:end, :);
% size(ax);
% size(sigmoidGradient(z2));
E2 = ax.*sigmoidGradient(z2)';
% size(E2);

if lambda ~= 0  
    Theta1_grad = (1/m)*(E2*entrada') +(lambda/m)*Theta1;
    Theta2_grad =(1/m)*(E3*y2') + (lambda/m)*Theta2;
else
    Theta1_grad = (1/m)*(E2*entrada');
    Theta2_grad = (1/m)*(E3*y2');
end

% Não altere esta linha!!
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
