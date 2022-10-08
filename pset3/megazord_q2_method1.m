%%% Q2 EM Algorithm, updating CCPs from the data

clear
clc
load("dataassign3.mat")

% firms are out of the market at the start
LagFirm1 = [zeros(5000, 1), Firm1(:, 1:4)];
StackFirm1 = reshape(Firm1, [], 1);
StackLagFirm1 = reshape(LagFirm1, [], 1);
StackState = reshape(State, [], 1);
% our starting point are the estimates from number 1

alpha_hat = [-0.49; -1.00; 0.31; -1.58];
gamma_hat = [6.98; 1.02; -0.28; -0.70];
sigma_hat = 0.97;

params = [alpha_hat;gamma_hat;sigma_hat];
params0 = ones(9,1);

% the initial guess for CCPs and the distribution of PState are taken from
% Question 1, too

p_101 = 0.56;
p_111 = 0.36;
p_001 = 0.23;
p_011 = 0.10;

pi_PState0 = 1-mean(PState);

% discount factor
beta = 0.9;

% state transitions can be taken from the data
state_transition = readtable('transitionState_allObserved.csv');

pi_00 = state_transition.transition_prob(state_transition.state_lag == 0 & state_transition.state_current == 0);
pi_01 = state_transition.transition_prob(state_transition.state_lag == 0 & state_transition.state_current == 1);
pi_10 = state_transition.transition_prob(state_transition.state_lag == 1 & state_transition.state_current == 0);
pi_11 = state_transition.transition_prob(state_transition.state_lag == 1 & state_transition.state_current == 1);

clear state_transition

count = 1;
% EM algorithm
while max(abs(params-params0))>1e-3 && count <= 200

    alpha = params(1:4);
    gamma = params(5:end-1);
    sigma = params(end);


    %We have four useful future value terms
    %Perm state==1, State==1
    V11= - pi_10*log(p_011) - pi_11*log(p_111);
    %Perm state==0, State==1
    V10= - pi_10*log(p_001) - pi_11*log(p_101);
    %Perm state==1, State==0
    V01= - pi_00*log(p_011) - pi_01*log(p_111);
    %Perm state==0, State==0
    V00= - pi_00*log(p_001) - pi_01*log(p_101);

    % flow payoffs
    u = zeros(5000, 5, 2);

    u(:, :, 1) = alpha(1) +alpha(2)*State + alpha(4)*(1-LagFirm1);
    u(:, :, 2) = alpha(1) +alpha(2)*State + alpha(3) + alpha(4)*(1-LagFirm1);

    % conditional valuation function:
    v = zeros(5000, 5, 2);
    % for PState = 0
    v(:, :, 1) = u(:, :, 1) + beta*(V10.*(State == 1) + V00.*(State==0)) + beta*0.577;
    % for PState = 1
    v(:, :, 2) = u(:, :, 2) + beta*(V11.*(State == 1) + V01.*(State==0)) + beta*0.577;

    % likelihood associated to each firm decision, considering PState = 0 or 1
    % (curly L_nst), we hit them with the choices actually being made
    l_nst = exp(v)./(1 + exp(v));
    like_firm_t = (l_nst.^(Firm1 == 1).*(1-l_nst).^(Firm1 == 0));
    % likelihood associated to price process - gives us identifying power

    likePrice_nst = zeros(5000, 5, 2);
    likePrice_nst(:, :, 1) = normpdf(Y - gamma(1) - gamma(2).*State - gamma(4).*Firm1, 0, sigma);
    likePrice_nst(:, :, 2) = normpdf(Y - gamma(1) - gamma(2).*State - gamma(3) - gamma(4).*Firm1, 0, sigma);

    % now we calculate the q's - they are a function of pi_PState and the l_nst
    like_nst = like_firm_t.*likePrice_nst;

    q_0 = pi_PState0.*prod(like_nst(:, :, 1), 2)./(pi_PState0.*prod(like_nst(:, :, 1), 2) ...
        + (1-pi_PState0).*prod(like_nst(:, :, 2), 2));

    q_1 = 1-q_0;

    % update pi_PState1
    pi_PState0 = mean(q_0);

    % Now we maximize the likelihood using q_1 as weights
    opt = fminunc(@(params)likelihood_q2(params(1:4), params(5:8), params(9), V00, V01, V10, V11, q_1, Firm1, State, Y), [alpha_hat;gamma_hat;sigma_hat]);

    params0 = params;
    params = opt;

    % update CCPs using the data
    q_1_stack = repmat(q_1, 5,1);
    q_0_stack = repmat(q_0, 5,1);

    p_011 = sum(q_1_stack.*(StackFirm1 == 0).*(StackState == 0).*(StackLagFirm1 == 1))./sum(q_1_stack.*(StackState == 0).*(StackLagFirm1 == 1));
    p_111 = sum(q_1_stack.*(StackFirm1 == 0).*(StackState == 1).*(StackLagFirm1 == 1))./sum(q_1_stack.*(StackState == 1).*(StackLagFirm1 == 1));
    p_001 = sum(q_0_stack.*(StackFirm1 == 0).*(StackState == 0).*(StackLagFirm1 == 1))./sum(q_0_stack.*(StackState == 0).*(StackLagFirm1 == 1));
    p_101 = sum(q_0_stack.*(StackFirm1 == 0).*(StackState == 1).*(StackLagFirm1 == 1))./sum(q_0_stack.*(StackState == 1).*(StackLagFirm1 == 1));

    count = count+1;
disp(["count:", count ])

end
