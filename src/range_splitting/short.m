b = 1.4 ;               
sigma = 1;            
M1 = -101;              
M2 = -1000000;              

sum_val = 0;
for ell = M2:M1
    sum_val = sum_val + 2 * pi * sigma * b^(ell);
end

result = 2 * log(b) * sum_val;

fprintf(result);
