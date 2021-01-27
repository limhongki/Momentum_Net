function u = func_bpg_m_diffld_v0_x0(A, u, u_old, W, y, M_f, u_0, mask, ldvec, w_u, delta)

E_u = w_u * delta;
u_p = u + E_u.*(u-u_old);

xi = u_p - div0( A'*(W.*( A*u_p - y )), M_f );

ldu0 = zeros(size(u_0));
ldI = zeros(size(u_0));
for l = 1:size(ldvec)
    %each slice corresponds to each 2d training/testing image
    ldu0(:,:,l) = ldvec(l) .* u_0(:,:,l);
    ldI(:,:,l) = ldvec(l) * mask;
end

u = div0(M_f .* xi + ldu0, M_f + ldI);
u = max(u, 0);



