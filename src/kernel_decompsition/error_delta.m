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
    % 输入：n是要求双阶乘的整数
    % 输出：n!!的结果
    
    result = 1;  % 初始化结果为1
    
    % 如果n为负数，则双阶乘没有定义，返回错误
    if n < 0
        error('双阶乘仅对非负整数有效');
    end
    
    % 迭代计算双阶乘
    while n > 0
        result = result * n;
        n = n - 2;  % 每次减去2
    end
end

