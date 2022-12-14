clear
clc
tic 
load('dataassign21.mat')

%Optimization
Params_q1= @(Params) q1(LY1,Y, X1t, X1, Params(1:5),Params(6:10),Params(11:15), Params(16),Params(17));
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-8,'OptimalityTolerance',10^-8);

S=fminunc(Params_q1,ones(17,1),OptimOptions);
toc
%% Backward recursion plus MLE 
function ll = q1(LY1, Y, X1t, X1, delta, gamma1,gamma2, beta,c)
    % computing flow utility net switching cost and error term on each counterfactual path
    u_counter = zeros(3000, 10, 5);
    
    for t=1:10
        for j=2:5
            u_counter(:,t, j)= X1t(:,t).*delta(j)+ X1(:,1).*gamma1(j)+ X1(:,2).*gamma2(j);
        end
    end
    
    %v=(individual, period, state, choice)
    v = zeros(3000, 10, 5, 5);
    
    % In the last period v is just u_counter (and a potential switching cost)
    for j=1:5
        for s=1:5
            v(:,10, s,j)=u_counter(:,10, j)-c*(s~=j);
        end
    end
    
    for t=9:-1:1
        %Decision states
        for s=2:5
            %Non terminal choices
            for j=2:5
                %Immediate U+expected U(taken choice as state) minus transition cost
                v(:, t, s, j) = u_counter(:,t, j) + beta.*(log(sum(exp(v(:, t+1,j,:)), 4))+0.57721)-c*(s~=j);
            end
            %Terminal choice
            v(:, t, s, 1) = u_counter(:,t, 1) + beta.*v(:,t+1,1,1)-c;
        end
        %Terminal state
        v(:, t, 1, 1) = u_counter(:,t, 1) + beta.*v(:,t+1,1,1);
    end
    
    %Denominator for every -individual-time-state combination
    denom = sum(exp(v(:, :, :,:)), 4);
    p = exp(v)./denom;

    % state Y=0 is absorbing
    p(:,:,1,1)=1;
    p(:,:,1,2:5)=0;

    % likelihood
    Y_choice= zeros(3000, 10, 5,5);
   
    Y_choice(:, 1, s,j) = (Y(:,1) ==j-1 & LY1==s-1);

    for t=2:10
        for  j=1:5
            for s=1:5
                Y_choice(:, t, s,j) = (Y(:,t) ==j-1 & Y(:,t-1)==s-1);
            end    
        end
    end

    ll=-sum(log(p(Y_choice==1)), "all");
end