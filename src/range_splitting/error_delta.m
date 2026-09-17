clear; clc; close all;
format long;
N = 500;
l = -12;
b = 1.4;
% delta = b^(2 * l);
delta = 0.5 / 800;
error_max = delta^2 / double_factorial(2);
error = zeros(N,1);
for i = 1:N
    error(i) = delta^(i+1) / double_factorial(2*i);
end

Error = sum(error);

function result = double_factorial(n)

    result = 1; 

    while n > 0
        result = result * n;
        n = n - 2;  
    end
end

