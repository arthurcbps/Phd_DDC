

load('dataassign3')
%Get state transitions and CCPs

CCPs=readtable('ccp_allObserved.csv');
state_transition = readtable('transitionState_allObserved.csv');

% Set discount factor
beta = 0.9;

%We need 4 CCPs-all combinations of State-Pstate fixing the current choice
%to be 0, the past choice to being 1: p_0(s, ps, 1)

p_011 = CCPs.CCP(CCPs.choice_current == 0 & CCPs.state_current == 0 & CCPs.Pstate == 1 & CCPs.choice_lag == 1);
p_111 = CCPs.CCP(CCPs.choice_current == 0 & CCPs.state_current == 1 & CCPs.Pstate == 1 & CCPs.choice_lag == 1);
p_001 = CCPs.CCP(CCPs.choice_current == 0 & CCPs.state_current == 0 & CCPs.Pstate == 0 & CCPs.choice_lag == 1);
p_101 = CCPs.CCP(CCPs.choice_current == 0 & CCPs.state_current == 1 & CCPs.Pstate == 0 & CCPs.choice_lag == 1);

pi_00 = state_transition.transition_prob(state_transition.state_lag == 0 & state_transition.state_current == 0);
pi_01 = state_transition.transition_prob(state_transition.state_lag == 0 & state_transition.state_current == 1);
pi_10 = state_transition.transition_prob(state_transition.state_lag == 1 & state_transition.state_current == 0);
pi_11 = state_transition.transition_prob(state_transition.state_lag == 1 & state_transition.state_current == 1);



%We have four useful future value terms
%Perm state==1, State==1
V11= - pi_10*log(p_011) - pi_11*log(p_111);
%Perm state==0, State==1
V10= - pi_10*log(p_001) - pi_11*log(p_101);
%Perm state==1, State==0
V01= - pi_00*log(p_011) - pi_01*log(p_111);
%Perm state==0, State==0
V00= - pi_00*log(p_001) - pi_01*log(p_101);

Params_LL_q1= @(Params) q1_logitLL(Params, State, PState, Firm1, V00, V11, V01, V10);
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-8,'OptimalityTolerance',10^-8);

S=fminunc(Params_LL_q1,ones(4,1),OptimOptions);


