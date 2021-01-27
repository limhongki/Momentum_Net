function u0 = func_mapping( IMout, Irecon_old, rho )

L = size(IMout,3);
u0 = zeros(size(IMout));
for l = 1:L
    if length(rho) == 1
        u0(:,:,l) = (1-rho)*Irecon_old(:,:,l) + rho*IMout(:,:,l);
    else
        u0(:,:,l) = (1-rho(l))*Irecon_old(:,:,l) + rho(l)*IMout(:,:,l);
    end
end


