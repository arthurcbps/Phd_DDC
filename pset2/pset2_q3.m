%% Question 3

clear
clc

load('dataassign23.mat')

T=10000;
rho = 0.05;
%% Nature's move arrival
% need transition from 0 to 1, 1 to 0 - just relative frequencies

mt_diff = diff(mt);
aux0 = find(State == 0);
aux0(end) = [];
time_in_state_0 = sum(mt_diff(aux0));
aux1 = find(State == 1);
time_in_state_1 = sum(mt_diff(aux1));




q_10_hat = sum(State == 1 & Naturemove == 1)/time_in_state_1;
q_11_hat = 1 - q_10_hat;
q_01_hat = sum(State == 0 & Naturemove == 1)/time_in_state_0;
q_00_hat = 1 - q_01_hat;
%% Player's move arrival
% That is just the relative frequency of being allowed to change
player_moved = 1-Naturemove;
lambda_hat = sum(player_moved)/10000;
clear player_moved aux0 aux1 time_in_state_0 time_in_state_1



%% Counting ocurrences of each state-action-incumbency status
% combinations are sjk - where s means incumbency status, j is the action and
% k is nature's state

% lagged incumbency, at time 0 s =0
lagIState=[0;IState(1:end-1)];

%

Y_111 = sum(Naturemove == 0 & IState == 1 & lagIState == 1 & State == 1);
Y_110 = sum(Naturemove == 0 & IState == 1 & lagIState == 1 & State == 0);
Y_101 = sum(Naturemove == 0 & IState == 1 & lagIState == 0 & State == 1);
Y_100 = sum(Naturemove == 0 & IState == 1 & lagIState == 0 & State == 0);
Y_011 = sum(Naturemove == 0 & IState == 0 & lagIState == 1 & State == 1);
Y_010 = sum(Naturemove == 0 & IState == 0 & lagIState == 1 & State == 0);
Y_001 = sum(Naturemove == 0 & IState == 0 & lagIState == 0 & State == 1);
Y_000 = sum(Naturemove == 0 & IState == 0 & lagIState == 0 & State == 0);

Y = [Y_111; Y_110 ; Y_101 ; Y_100 ; Y_011; Y_010 ; Y_001 ; Y_000];

%%

%Optimization
Params_q3= @(Params) continuous_ll(Y, lambda_hat, rho, q_01_hat,q_10_hat,Params(1), Params(2),Params(3));
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-8,'OptimalityTolerance',10^-8);

S=fminunc(Params_q3,zeros(3,1),OptimOptions);
toc

%% Nested fixed point
% parameters to be estimated are flow payoffs (vector u) that are state varying,
% and a switching cost paid by those who choose to enter

function neglike = continuous_ll(crosstab, lambda, rho, q_01, q_10, u_0, u_1, c)

u = [u_0 u_1]';
q = [q_01 q_10]';
V = zeros(2, 1);
denom_V = lambda + rho + q;

% iteration parameters
tol = 1e-6;
count = 0;
diff = 1;


while count < 1000 && diff > tol
oldV= V;
% Note that we do not even need to use the switching costs to compute the
% value function - the contraction mapping theorem plus the known form of
% the expectation of the maximum of logit errors already take care of
% computing V's

% Also note that these do not depend on j!! j only enter as flow payoffs
% when we form the ll
newV = (u + [oldV(2); oldV(1)].*q + lambda*log(exp(oldV)+1) + lambda*0.5772)./denom_V;

count = count+1;
diff = max(abs(newV -oldV));
V = newV;
end

% forming the likelihood - we need to consider eight different conditional
% probabilities - 2 for each state times 2 for each action (ie paying
% switching cost to change or not), 2 for each individual state (being on the market or not)
% These have the usual logit form

% Notation is P_sjk - where s means incumbency status, j is the action and
% k is nature's state

P_111 = exp(V(2))/(1+exp(V(2)));
P_110 = exp(V(1))/(1+exp(V(1)));
P_101 = 1 - P_111;
P_100 = 1 - P_110;
P_011 = exp(V(2) - c)/(1 + exp(V(2)-c));
P_010 = exp(V(1) - c)/(1 + exp(V(1)-c));
P_001 = 1 - P_011;
P_000 = 1 - P_010;

P = [P_111; P_110 ; P_101 ; P_100 ; P_011; P_010 ; P_001 ; P_000];

neglike = -crosstab'*log(P);

end 
