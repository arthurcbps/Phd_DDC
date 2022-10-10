function ll=likelihood_q4_2ndStage(alpha, PState, Firm1, V00, V11, V01, V10, q_1)
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
like_nst = (l_nst.^(Firm1 == 1).*(1-l_nst).^(Firm1 == 0));



ll = -sum(q_1.*log(like_nst(:,:,2)) + (1-q_1).*log(like_nst(:,:,1)), "all");

end