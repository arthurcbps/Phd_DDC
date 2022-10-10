function ll = likelihood_q4_1stStage(gamma, sigma, q_1, Firm1, PState, Y)

likePrice_nst = zeros(5000, 5, 2);
likePrice_nst(:, :, 1) = normpdf(Y - gamma(1) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);
likePrice_nst(:, :, 2) = normpdf(Y - gamma(1) - gamma(2) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);




ll = -sum(q_1.*log(likePrice_nst(:,:,2)) + (1-q_1).*log(likePrice_nst(:,:,1)), "all");
end





