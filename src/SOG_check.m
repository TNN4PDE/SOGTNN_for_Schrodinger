b  = 1.22749083347315613;          
sigma = 1;       
ell = -9:150;       
r_c = 15;
r   = linspace(-r_c,r_c,1024);
[R1,R2] = meshgrid(r,r);
rho = sqrt(1 + (R1 - R2).^2);   

ell3  = reshape(ell,1,1,[]);    % 1×1×Length
bEll  = b.^ell3;                % 1×1×Length
term  = (1/bEll) .* exp(-0.5/(bEll*sigma).^2) .* exp(-0.5*((R1 - R2)./(bEll*sigma)).^2);  % 400×400×Length
approx = sum(term,3);          
normalization_constant = (2 * log(b)) / sqrt(2 * pi * sigma^2);
approx = normalization_constant * approx;
figure('Color','w');
subplot(2,2,1)
surf(R1,R2,1./rho,'EdgeColor','none')
title('Original f(r_1,r_2)')
zlabel('f');
shading interp; colorbar

subplot(2,2,2)
surf(R1,R2,approx,'EdgeColor','none')
title('Gaussian-sum approximation')
zlabel('approx');
shading interp; colorbar

subplot(2,2,3)
surf(R1,R2,abs(approx - 1./rho),'EdgeColor','none')
title('Absolute error')
zlabel('|error|');
shading interp; colorbar

subplot(2,2,4)
surf(R1,R2,abs(approx.*rho - 1),'EdgeColor','none')
title('Absolute error')
zlabel('|error|');
shading interp; colorbar