function ll=q1_logitLL(alpha, State, PState, Firm1, V00, V11, V01, V10)
beta = 0.9;
LagFirm1 = [zeros(5000, 1), Firm1(:, 1:4)];

t=size(State,2);

% computing flow utility (market,lag choice,state) for those that choose to
% be in the market
u = alpha(1) +alpha(2)*State + alpha(3)*PState + alpha(4)*(1-LagFirm1);

%value function of entering/staying
aux = V11*(repelem(PState,1,t)==1).*(State==1) ...
+ V10*(repelem(PState,1,t)==0).*(State==1) ...
+ V01*(repelem(PState,1,t)==1).*(State==0) ...
+ V00*(repelem(PState,1,t)==0).*(State==0);

v = u+ beta*aux + beta*0.577;

ll=sum(log(exp(v)+1)-Firm1.*v, "all");

end