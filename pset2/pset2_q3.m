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




q_10 = sum(State == 1 & Naturemove == 1)/time_in_state_1;
q_11 = 1 - q_10;
q_01 = sum(State == 0 & Naturemove == 1)/time_in_state_0;
q_00 = 1 - q_01;
%% Player's move arrival
% That is just the relative frequency of being allowed to change
player_moved = 1-Naturemove;
lambda = sum(player_moved)/10000;

%% Nested fixed point
% only two states, initial guess is 0
% parameters to be estimated are flow payoffs (vector u) that are state varying,
% and a switching cost paid by those who choose to enter

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
disp(['Iteration no. ' int2str(count), ' max val fn diff is ' num2str(diff)]);
end

% forming the likelihood - we need to consider four different conditional
% probabilities - 2 for each state times 2 for each action (ie paying
% switching cost to change or not). These have the usual logit form

P_Vc= exp(V - c)./(1 + exp(V-c));
P_Vnc = exp(V)./(1 + exp(V));

P = [P_Vc, P_Vnc]
