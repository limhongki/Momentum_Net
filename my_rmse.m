function rmse = my_rmse(I, ref, ig)

% rmse = sqrt( ( sum((I(ig.mask) - ref(ig.mask)).^2) ) / sum(ig.mask(:)) );
rmse = norm(I(ig.mask)-ref(ig.mask))/sqrt(sum(ig.mask(:)));
end