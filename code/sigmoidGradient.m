function g = sigmoidGradient(z)
%SIGMOIDGRADIENT devolve o gradiente da função sigmoid no ponto z
% z pode ser uma matriz, um vetor ou um número real

g = zeros(size(z));

g  = sigmoid(z).*(1-sigmoid(z));



end
