function ll = likelihood_q3(alpha, gamma, sigma, V00, V01, V10, V11, q_1, Firm1, PState, Y)
beta = 0.9;
LagFirm1 = [zeros(5000, 1), Firm1(:, 1:4)];

% Flow payoffs in each case
u(:, :, 1) = alpha(1) + alpha(3).*PState + alpha(4).*(1-LagFirm1);
u(:, :, 2) = alpha(1) + alpha(2) + alpha(3).*PState + alpha(4).*(1-LagFirm1);

% conditional valuation function:
v = zeros(5000, 5, 2);
% for State = 0
v(:, :, 1) = u(:, :, 1) + beta*(V01.*(PState == 1) + V00.*(PState==0)) + beta*0.577;
% for State = 1
v(:, :, 2) = u(:, :, 2) + beta*(V11.*(PState == 1) + V10.*(PState==0)) + beta*0.577;

% likelihood associated to each firm decision, considering PState = 0 or 1
% (curly L_nst), we hit them with the choices actually being made
l_nst = exp(v)./(1 + exp(v));
like_firm_t = (l_nst.^(Firm1 == 1).*(1-l_nst).^(Firm1 == 0));
% likelihood associated to price process - gives us identifying power

likePrice_nst = zeros(5000, 5, 2);
likePrice_nst(:, :, 1) = normpdf(Y - gamma(1) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);
likePrice_nst(:, :, 2) = normpdf(Y - gamma(1) - gamma(2) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);


% joint likelihood for a given n-by-t-by-s
like_nst = like_firm_t.*likePrice_nst;



ll = -sum(q_1.*log(like_nst(:,:,2)) + (1-q_1).*log(like_nst(:,:,1)), "all");
end





