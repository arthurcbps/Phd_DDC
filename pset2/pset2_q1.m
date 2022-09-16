clear
clc
load('dataassign22.mat')


%%
% Normalization
X1 = X1(:, 2) - X1(:,1);

Params_q1_no_switch= @(Params) q1_no_switch(Y, X1t, X1, Y_lag, Params(1:5),Params(6:10), Params(11));

%% 
% A draft on using switching costs
% fixing parameter values just to make it simpler
gamma = [0, -1, -2, -3, -4];
delta = [0, 1, 2, 3, 4];
beta = 0.95;
c = 5;
u = zeros(5000,10);

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

t = 9;
while t > 0
    j = 0;
    while j<= 4
        v(:, t, j+1) = u_counter(:,t, j+1) + beta.*(log(sum(exp(v(:, t+1,:)), 3))+0.57721);
        j = j+1;
    end
    t =t-1;
end

denom = sum(exp(v(:, :, :)), 3);
p = exp(v)./denom;

% likelihood
Y_choice= zeros(5000, 10, 5);
j=0;
while j<=4
    Y_choice(:, :, j+1) = (Y ==j);
    j = j+1;
end
ll=-sum(Y_choice.*log(p), "all");
end
        

