
function [p_011, p_111, p_001, p_101] = update_ccp_data_q2(q_011, q_111, q_001, q_101, Firm1, State, LagFirm1)
% q's are guesses for conditional type probabilities where q_jkl is such
% that j = state, k = Pstate, l = lag firm choice, all considering 
% current firm choice being 0 - PState treated as unobserved

Firm1 = reshape(Firm1, [], 1);
LagFirm1 = reshape(LagFirm1, [], 1);
State = reshape(State, [], 1);
q_011 = repmat(q_011, 5,1);
q_111 = repmat(q_111, 5,1);
q_001 = repmat(q_001, 5,1);
q_101 = repmat(q_101, 5,1);


p_011 = sum(q_011.*(Firm1 == 0).*(State == 0).*(LagFirm1 == 1))./sum(q_011.*(State == 0).*(LagFirm1 == 1));
p_111 = sum(q_111.*(Firm1 == 0).*(State == 1).*(LagFirm1 == 1))./sum(q_111.*(State == 1).*(LagFirm1 == 1));
p_001 = sum(q_001.*(Firm1 == 0).*(State == 0).*(LagFirm1 == 1))./sum(q_001.*(State == 0).*(LagFirm1 == 1));
p_101 = sum(q_101.*(Firm1 == 0).*(State == 1).*(LagFirm1 == 1))./sum(q_101.*(State == 1).*(LagFirm1 == 1));

end
