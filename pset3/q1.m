load('dataassign3')

CCPs=readtable('ccp_allObserved.csv');
state_transition = readtable('transitionState_allObserved.csv');
%Get state transitions
state_transition = table2array(state_transition);

LState=State(:,1:4);
FState=State(:,2:5);

p11_1=sum(LState==1 & FState==1 & repelem(PState,1,4)==1 )/sum(LState==1 & repelem(PState,1,4)==1);
p11_0=sum(LState==1 & FState==1 & repelem(PState,1,4)==0 )/sum(LState==1 & repelem(PState,1,4)==0);

p00_1=sum(LState==0 & FState==0 & repelem(PState,1,4)==1 )/sum(LState==0 & repelem(PState,1,4)==1);
p00_0=sum(LState==0 & FState==0 & repelem(PState,1,4)==0 )/sum(LState==0 & repelem(PState,1,4)==0);

%Restrict to relevant CCPs
CCPs=CCPs.CCP(CCPs.choice_current==1 & CCPs.choice_lag==1,:);

%Get useful future values
%Perm state==1, State==1
V11=(p11_1*log(CCPs(4))+(1-p11_1)*log(CCPs(2)));
%Perm state==1, State==0
V10=((1-p00_1)*log(CCPs(4))+p00_1*log(CCPs(2)));
%Perm state==0, State==1
V01=(p11_1*log(CCPs(3))+(1-p11_1)*log(CCPs(1)));
%Perm state==0, State==0
V00=((1-p00_0)*log(CCPs(3))+p00_0*log(CCPs(1)));

v(PState, State, Firm1, .9, ones(1,4),V11,V10,V01,V00)

%%  MLE 
function v = v(PState, State, Firm1, beta, alpha,V11,V10,V01,V00)
    
    t=size(State,2);
    n=size(State,1);

    % computing flow utility (market,choice,state)
    
    u=alpha(1)*ones(n,1)+alpha(2)*State+alpha(3)*repelem(PState,1,t)+alpha(4)*(ones(size(Firm1))-Firm1);
    
   %value function of entering/staying
   aux=V11*(repelem(PState,1,t)==1).*(State==1)+V10*(repelem(PState,1,t)==1).*(State==0)+V01*(repelem(PState,1,t)==0).*(State==1)+V00*(repelem(PState,1,t)==0).*(State==0);
   v=u;
   v(:,1:4)=v(:,1:4)+beta*aux(:,1:4);

end
