%%% Q3 EM Algorithm,

clear
clc
load("dataassign3.mat")
clear State

% firms are out of the market at the start
LagFirm1 = [zeros(5000, 1), Firm1(:, 1:4)];
StackPState = repmat(PState, 1, 5);
% our starting point are the estimates from number 1

alpha_hat = [-0.49; -1.00; 0.31; -1.58];
gamma_hat = [6.98; 1.02; -0.28; -0.70];
sigma_hat = 0.97;

params = [alpha_hat;gamma_hat;sigma_hat];
params0 = ones(9,1);

% discount factor
beta = 0.9;


% the initial guess for CCPs and the state transition are taken from
% Question 1, too (adding a bit of noise)

p_101 = 0.5;
p_111 = 0.3;
p_001 = 0.2;
p_011 = 0.10;


pi_00 = 0.85;
pi_01 = 0.15;
pi_10 = 0.4;
pi_11 = 0.67;

pi = [pi_00 pi_01; pi_10 pi_11];

% We also take from the data an initial guess of the initial distribution
% of the state  - mean(State(:, 1));

initial_pi = [0.8 0.2];

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

    % flow payoffs - considering State = 0, and State = 1
    u = zeros(5000, 5, 2);

    u(:, :, 1) = alpha(1) + alpha(3).*PState + alpha(4).*(1-LagFirm1);
    u(:, :, 2) = alpha(1) + alpha(2) + alpha(3).*PState + alpha(4).*(1-LagFirm1);

    % conditional valuation function:
    v = zeros(5000, 5, 2);
    % for State = 0
    v(:, :, 1) = u(:, :, 1) + beta*(V01.*(PState == 1) + V00.*(PState==0)) + beta*0.577;
    % for State = 1
    v(:, :, 2) = u(:, :, 2) + beta*(V11.*(PState == 1) + V10.*(PState==0)) + beta*0.577;

    % likelihood associated to each firm decision, considering State = 0 or 1
    % (curly L_nst), we hit them with the choices actually being made
    l_nst = exp(v)./(1 + exp(v));
    like_firm_t = (l_nst.^(Firm1 == 1).*(1-l_nst).^(Firm1 == 0));
    % likelihood associated to price process - gives us identifying power

    likePrice_nst = zeros(5000, 5, 2);
    likePrice_nst(:, :, 1) = normpdf(Y - gamma(1) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);
    likePrice_nst(:, :, 2) = normpdf(Y - gamma(1) - gamma(2) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);


    % joint likelihood for a given n-by-t-by-s
    like_nst = like_firm_t.*likePrice_nst;

    % now we calculate the q's and update initial distribution of the state and its transitions
    % - they are a function of guessed initial distribution,
    % guessed transitions and the l_nst - Using past year's code (don't
    % know how to do that myself)

    [initial_pi, pi, q_n1t] = typeprob(initial_pi, pi, like_nst);
    q_n0t = 1-q_n1t;

    pi_00 = pi(1,1);
    pi_01 = pi(1,2);
    pi_10 = pi(2,1);
    pi_11 = pi(2,2);

    % Now we maximize the likelihood using q as weights
    opt = fminunc(@(params)likelihood_q3(params(1:4), params(5:8), params(9), V00, V01, V10, V11, q_n1t, Firm1, PState, Y), [alpha_hat;gamma_hat;sigma_hat]);

    params0 = params;
    params = opt;

    % update CCPs using the data, noting that now the q's vary over time


    p_011 = sum(q_n0t.*(Firm1 == 0).*(StackPState == 1).*(LagFirm1 == 1), "all")./sum(q_n0t.*(StackPState == 1).*(LagFirm1 == 1), "all");
    p_111 = sum(q_n1t.*(Firm1 == 0).*(StackPState == 1).*(LagFirm1 == 1), "all")./sum(q_n1t.*(StackPState == 1).*(LagFirm1 == 1), "all");
    p_001 = sum(q_n0t.*(Firm1 == 0).*(StackPState == 0).*(LagFirm1 == 1), "all")./sum(q_n0t.*(StackPState == 0).*(LagFirm1 == 1), "all");
    p_101 = sum(q_n1t.*(Firm1 == 0).*(StackPState == 0).*(LagFirm1 == 1), "all")./sum(q_n1t.*(StackPState == 0).*(LagFirm1 == 1), "all");

    count = count+1;
    disp(["count:", count ])

end
