function psnrval = my_psnr( I, ref, peakval, ig )

err = (norm(I(ig.mask)-ref(ig.mask),2).^2) / sum(ig.mask(:));

psnrval = 10*log10(peakval.^2/err);

end