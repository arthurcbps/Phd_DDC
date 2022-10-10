% Q4

% First stage - use EM algorithm to estimate gammas and sigma, note that we
% don't need to update CCPs here, we can use them only after the first
% stage

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

params = [gamma_hat;sigma_hat];
params0 = ones(5,1);

% discount factor
beta = 0.9;


% initial guess of state transitions
pi_00 = 0.85;
pi_01 = 0.15;
pi_10 = 0.4;
pi_11 = 0.6;

pi = [pi_00 pi_01; pi_10 pi_11];

% We also take from the data an initial guess of the initial distribution
% of the state  - mean(State(:, 1));

initial_pi = [0.8 0.2];


count = 1;
while max(abs(params-params0))>1e-3 && count <= 200

    gamma = params(1:4);
    sigma = params(5);

    % likelihood associated to price process

    likePrice_nst = zeros(5000, 5, 2);
    likePrice_nst(:, :, 1) = normpdf(Y - gamma(1) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);
    likePrice_nst(:, :, 2) = normpdf(Y - gamma(1) - gamma(2) - gamma(3).*PState - gamma(4).*Firm1, 0, sigma);

    % now we calculate the q's and update initial distribution of the state and its transitions
    % - they are a function of guessed initial distribution,
    % guessed transitions and the l_nst - Using past year's code (don't
    % know how to do that myself)

    [initial_pi, pi, q_n1t] = typeprob(initial_pi, pi, likePrice_nst);
    q_n0t = 1-q_n1t;

    pi_00 = pi(1,1);
    pi_01 = pi(1,2);
    pi_10 = pi(2,1);
    pi_11 = pi(2,2);

    % Nowe take these q's and maximize the likelihood

    opt_price = fminunc(@(params)likelihood_q4_1stStage(params(1:4), params(5), q_n1t, Firm1, PState, Y), [gamma_hat;sigma_hat]);

    params0 = params;
    params = opt_price;

    count = count+1;
    disp(["count:", count ])

end

% Note how the gamma's are a bit far away from the truth - makes sense,
% given we are only the price process to identify it, instead of firm's
% choices too

% Get CCps from the data using estimated q's
p_011 = sum(q_n0t.*(Firm1 == 0).*(StackPState == 1).*(LagFirm1 == 1), "all")./sum(q_n0t.*(StackPState == 1).*(LagFirm1 == 1), "all");
p_111 = sum(q_n1t.*(Firm1 == 0).*(StackPState == 1).*(LagFirm1 == 1), "all")./sum(q_n1t.*(StackPState == 1).*(LagFirm1 == 1), "all");
p_001 = sum(q_n0t.*(Firm1 == 0).*(StackPState == 0).*(LagFirm1 == 1), "all")./sum(q_n0t.*(StackPState == 0).*(LagFirm1 == 1), "all");
p_101 = sum(q_n1t.*(Firm1 == 0).*(StackPState == 0).*(LagFirm1 == 1), "all")./sum(q_n1t.*(StackPState == 0).*(LagFirm1 == 1), "all");


% Compute future value terms and flow utility, as before:

%We have four useful future value terms
    %Perm state==1, State==1
    V11= - pi_10*log(p_011) - pi_11*log(p_111);
    %Perm state==0, State==1
    V10= - pi_10*log(p_001) - pi_11*log(p_101);
    %Perm state==1, State==0
    V01= - pi_00*log(p_011) - pi_01*log(p_111);
    %Perm state==0, State==0
    V00= - pi_00*log(p_001) - pi_01*log(p_101);

opt_firm = fminunc(@(params)likelihood_q4_2ndStage(params(1:4), PState, Firm1, V00, V11, V01, V10, q_n1t), alpha_hat);

% much more imprecise
