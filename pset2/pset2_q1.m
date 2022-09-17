clear
clc
load('dataassign22.mat')


%%
% Normalization
X1 = X1(:, 2) - X1(:,1);

Params_q1= @(Params) q1(LY1,Y, X1t, X1, Params(1:5),Params(6:10), Params(11),Params(12));

%% 
% fixing parameter values just to make it simpler
gamma = [0, -1, -2, -3, -4];
delta = [0, 1, 2, 3, 4];
beta = 0.95;
c = 5;
u = zeros(5000,10);

%Evaluate log-likelihood
Params_q1([delta,gamma,beta,c])

% computing observed flow utility net of switching costs and error term
for t=1:10
    j=0;
    while j<= 4
        u(:,t)=(Y(:,t)==j).*(X1t(:,t).*delta(j+1)+X1.*gamma(j+1))+u(:,t);
        j=j+1;
    end
end

% compute observed flow utility with switching costs net of the error term
u_minus_c = zeros(5000, 10);
u_minus_c(:,1) = u(:,1) - (LY1(:,1) ~= Y(:,1)).*c;
u_minus_c(:,2:10) = u(:,2:10) - (Y(:, 1:9) ~= Y(:, 2:10)).*c;

<<<<<<< Updated upstream
%% Backward recursion plus MLE - assume no switching costs
function ll = q1_no_switch(Y, X1t, X1, delta, gamma, beta)
% computing flow utility net and error term on each counterfactual path

u_counter = zeros(5000, 10, 5);
for t=1:10
    j=0;
    while j<= 4
        u_counter(:,t, j+1)= X1t(:,t).*delta(j+1)+ X1.*gamma(j+1);
        j=j+1;
    end
end

v = zeros(5000, 10, 5);
% recall that v has no future term in the last period, so in the last
% period it is just u_counter
j=0;
while j<= 4
    v(:,10, j+1)=(X1t(:,10).*delta(j+1)+X1.*gamma(j+1));
    j=j+1;
end
=======

%% Backward recursion plus MLE 
function ll = q1(LY1, Y, X1t, X1, delta, gamma, beta,c)
>>>>>>> Stashed changes

    % computing flow utility net switching cost and error term on each counterfactual path
    u_counter = zeros(5000, 10, 5);
    for t=1:10
        for j=1:5
            u_counter(:,t, j)= X1t(:,t).*delta(j)+ X1.*gamma(j);
        end
    end
    
    %v=(individual, period, state, choice)
    v = zeros(5000, 10, 5, 5);
    
    % In the last period v is just u_counter (and a potential switching cost)
    for j=1:5
        for s=1:5
            v(:,10, s,j)=u_counter(:,t, j)-c*(s~=j);
        end
    end
    
    for t=9:1
        for j=1:5
            for s=1:5
                %Immediate U+expected U(taken choice as state) minus transition cost
                v(:, t, s, j) = u_counter(:,t, j) + beta.*(log(sum(exp(v(:, t+1,j,:)), 4))+0.57721)-c*(s~=j);
            end
        end
    end
    
    %Denominator for every -individual-time-state combination
    denom = sum(exp(v(:, :, :,:)), 4);
    p = exp(v)./denom;
    
    % likelihood
    Y_choice= zeros(5000, 10, 5,5);
   
    Y_choice(:, 1, s,j) = (Y(:,1) ==j & LY1==s);

    for t=2:10
        for  j=1:5
            for s=1:5
                Y_choice(:, t, s,j) = (Y(:,t) ==j & Y(:,t-1)==s);
            end    
        end
    end
    ll=-sum(Y_choice.*log(p), "all");
end
        

