function ld_vec = func_ld_est_corr( M, condnum, mask, L )

if condnum < 1
   error('the desired condition number must be larger than 1.')
end

ld_vec = zeros(L,1);
for l=1:L
    Ml = M(:,:,l);
    Mvecl = col(Ml(mask));
    if min(Mvecl) >= 0
        ld_vec(l) = (max(Mvecl) - condnum*min(Mvecl)) / (condnum - 1);
    else
        error('the min value of elements in digonal majorizer must be larger than 0!');
    end
    
end

