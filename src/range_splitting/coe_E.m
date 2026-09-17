N = 25; 
b = 2.5;
l = -4;
delta = 2 * b^(2*l);
num_points = 5000;
M = N; 
nodes = cos((2*(0:num_points-1)+1) * pi / (2 * num_points));
x = linspace(-1, 1, num_points);
y = linspace(-1, 1, num_points);
[X, Y] = meshgrid(x, y);
gaussian_kernel = exp(-(X - Y).^2 / delta);
E = zeros(N + 1, N + 1);
for k = 0:N
    for l = 0:N
        if k + l <= M  
            if mod(k + l, 2) == 1
                E(k + 1, l + 1) = 0; 
            continue; 
            end
            integrand = @(x, y) exp(-(x - y).^2 / delta) .* chebyshev_poly(x, k) .* chebyshev_poly(y, l);
            
            integral_value = 0;
            for i = 1:num_points
                for j = 1:num_points
                    integral_value = integral_value + ((pi^2) / (num_points^2)) * integrand(nodes(i), nodes(j));
                end
            end

            if k == 0 && l == 0
                E(k + 1, l + 1) = (1 / pi^2) * integral_value;
            elseif k == 0
                E(k + 1, l + 1) = (2 / pi^2) * integral_value;
            elseif l == 0
                E(k + 1, l + 1) = (2 / pi^2) * integral_value;
            else
                E(k + 1, l + 1) = (4 / pi^2) * integral_value;
            end
        end
    end
end
approximation = zeros(size(X));
for k = 0:N
    for l = 0:N
        if k + l <= M  
            approximation = approximation + E(k + 1, l + 1) * chebyshev_poly(X, k) .* chebyshev_poly(Y, l);
        end
    end
end
error_surface = log10(abs(gaussian_kernel - approximation));
save('E_4.mat','E');
figure;
subplot(1, 2, 1);
surf(X, Y, gaussian_kernel, 'EdgeColor', 'none'); hold on;
surf(X, Y, approximation, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
xlabel('x');
ylabel('y');
zlabel('Value');
legend('Exact Gaussian Kernel', 'Chebyshev Approximation');
title('Two-Variate Chebyshev Expansion of Gaussian Kernel');
grid on;
subplot(1, 2, 2);
surf(X, Y, error_surface, 'EdgeColor', 'none');
xlabel('x');
ylabel('y');
zlabel('Error');
title('Error Surface: Exact - Approximation');
colorbar; 
grid on;
function Tn = chebyshev_poly(x, n)
    if n == 0
        Tn = ones(size(x)); % T_0(x) = 1
    elseif n == 1
        Tn = x; % T_1(x) = x
    else
        Tn_2 = ones(size(x)); % T_0
        Tn_1 = x; % T_1
        Tn = zeros(size(x));
        for k = 2:n
            Tn = 2 * x .* Tn_1 - Tn_2; 
            Tn_2 = Tn_1;
            Tn_1 = Tn; 
        end
    end
end

function w = weight(x)
    w = sqrt(1 - x.^2).^(-1);
end
