%% load
data_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
sData = load(fullfile(data_dir, 's_wbt_rc15_1e-12.mat'));
wData = load(fullfile(data_dir, 'w_wbt_rc15_1e-12.mat'));
s = sData.s_wbt;   
w = wData.w_wbt;
n=size(s,1);
w = reshape(w,1,1,n);         
s = reshape(s,1,1,n);         
r_c = 15;
numel = 1024;
r   = linspace(-r_c,r_c,numel);
[R1,R2] = meshgrid(r,r);
rho = sqrt(1 + (R1 - R2).^2);  
core = reshape((R1 - R2).^2,numel,numel,1); % → 400×400×1
approx = sum( w .* exp(-s .* core) , 3 );   % 400×400
max(max(abs(approx - 1./rho)))
max(max(abs(approx.*rho - 1)))
figure('Color','w');
subplot(2,2,1)
surf(R1,R2,1./rho,'EdgeColor','none')
title('Original f(r_1,r_2)')
xlabel('r_1'); ylabel('r_2'); zlabel('f');
shading interp; colorbar

subplot(2,2,2)
surf(R1,R2,approx,'EdgeColor','none')
title('Gaussian-sum approximation')
xlabel('r_1'); ylabel('r_2'); zlabel('approx');
shading interp; colorbar

subplot(2,2,3)
surf(R1,R2,abs(approx - 1./rho),'EdgeColor','none')
title('Absolute error')
xlabel('r_1'); ylabel('r_2'); zlabel('|error|');
shading interp; colorbar

subplot(2,2,4)
surf(R1,R2,abs(approx.*rho - 1),'EdgeColor','none')
title('Relative error')
xlabel('r_1'); ylabel('r_2'); zlabel('|error|');
shading interp; colorbar

