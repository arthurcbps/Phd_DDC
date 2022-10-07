

% Since PState is unobserved, create two 5000x5 arrays, one for PState=0
% the other for PState = 1
alpha = unifrnd(-1, 1, 4, 1);
u = zeros(5000, 5, 2);

u(:, :, 1) = alpha(1) +alpha(2)*State + alpha(4)*(1-LagFirm1);
u(:, :, 2) = alpha(1) +alpha(2)*State + alpha(3) + alpha(4)*(1-LagFirm1);


% use four guesses of useful ccps p_011, p_111, p_001, p_101
% also use State transitions, taken from the data - pi_00 pi_01 pi_10 pi_11

%We have four useful future value terms
%Perm state==1, State==1
V11= - pi_10*log(p_011) - pi_11*log(p_111);
%Perm state==0, State==1
V10= - pi_10*log(p_001) - pi_11*log(p_101);
%Perm state==1, State==0
V01= - pi_00*log(p_011) - pi_01*log(p_111);
%Perm state==0, State==0
V00= - pi_00*log(p_001) - pi_01*log(p_101);

% conditional valuation function:

v = zeros(5000, 5, 2);

% for PState = 0
v(:, :, 1) = u(:, :, 1) + beta*(V10.*(State == 1) + V00.*(State==0)) + beta*0.577;
% for PState = 1
v(:, :, 2) = u(:, :, 2) + beta*(V11.*(State == 1) + V01.*(State==0)) + beta*0.577;;

% likelihood associated to each observation, considering PState = 0 or 1 
% (curly L_nst)

l_nst = v./exp(1 + v);

% now we calculate the q's - they are a function of pi_PState and the l_nst

pi_PState = 0.5;

q_n0 = pi_PState.*l_nst(:,1,1).*l_nst(:,2,1).*l_nst(:,3,1).*l_nst(:,4,1).*l_nst(:,5,1)./...
    (pi_PState.*l_nst(:,1,1).*l_nst(:,2,1).*l_nst(:,3,1).*l_nst(:,4,1).*l_nst(:,5,1) ...
    + (1-pi_PState).*l_nst(:,1,2).*l_nst(:,2,2).*l_nst(:,3,2).*l_nst(:,4,2).*l_nst(:,5,2));

q_n1 = 1 - q_n0;



