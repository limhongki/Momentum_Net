function im_clean = autoEncCNNrot( im_noisy, W, D, alpha, K )

softshrink = @(u, theta) (abs(u) > exp(theta)) .* ((abs(u)-exp(theta)) .* sign(u));

im_clean = zeros(size(im_noisy));
for k= 1 : K
        
   %Convolution with zero-boundary condition
   %!!!NOTE: we compensate rotating filter in conv2().
   
   im_clean_kth = conv2( softshrink( conv2(im_noisy, rot90(W(:,:,k),2), 'same'), alpha(k) ), rot90(D(:,:,k),2), 'same' );
   im_clean = im_clean + im_clean_kth;
    
end

end