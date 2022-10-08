function ll = likelihood_q2(alpha, gamma, sigma, V00, V01, V10, V11, q_1, Firm1, State, Y)
beta = 0.9;
LagFirm1 = [zeros(5000, 1), Firm1(:, 1:4)];
% Since PState is unobserved, create two 5000x5 arrays, one for PState=0
% the other for PState = 1

u = zeros(5000, 5, 2);

u(:, :, 1) = alpha(1) +alpha(2)*State + alpha(4)*(1-LagFirm1);
u(:, :, 2) = alpha(1) +alpha(2)*State + alpha(3) + alpha(4)*(1-LagFirm1);


% conditional valuation function:

v = zeros(5000, 5, 2);

% for PState = 0
v(:, :, 1) = u(:, :, 1) + beta*(V10.*(State == 1) + V00.*(State==0)) + beta*0.577;
% for PState = 1
v(:, :, 2) = u(:, :, 2) + beta*(V11.*(State == 1) + V01.*(State==0)) + beta*0.577;

% part of the likelihood associated to each market decision, considering PState = 0 or 1 
% (curly L_nst)

l_nst = exp(v)./(1 + exp(v));


% part of the likelihood related to the price process - one assuming PState
% = 0, the other PState = 1, we will weight them by q_1

likePrice_nst = zeros(5000, 5, 2);
 likePrice_nst(:, :, 1) = normpdf(Y - gamma(1) - gamma(2).*State - gamma(4).*Firm1, 0, sigma);
    likePrice_nst(:, :, 2) = normpdf(Y - gamma(1) - gamma(2).*State - gamma(3) - gamma(4).*Firm1, 0, sigma);

ll = -sum(q_1.*(log(l_nst(:, :, 2)).*(Firm1 == 1) + log(1-l_nst(:, :, 2)).*(Firm1 == 0) + log(likePrice_nst(:, :, 2))) ...
    + (1-q_1).*(log(l_nst(:, :, 1)).*(Firm1 == 1) + log(1-l_nst(:, :, 1)).*(Firm1 == 0) + log(likePrice_nst(:, :, 1))), "all");
end





