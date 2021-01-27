function [delta] = func_Delta_est( Irecon_old, I0_old, I0, L )

delta = zeros(1,L);
for l = 1:L
    delta(l) = norm(col(Irecon_old(:,:,l) - I0(:,:,l)), 2)^2 ...
        - norm(col(Irecon_old(:,:,l) - I0_old(:,:,l)), 2)^2;
end

end

