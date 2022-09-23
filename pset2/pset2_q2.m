clear
%% 
%Reading cleaned data from R (data with appropriate sample and in long
%format - each row is from individual i at period t


opts = delimitedTextImportOptions("NumVariables", 9);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["id", "time", "X_present", "X_past", "Y_past", "Y1", "Y2", "Y3", "Y4"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Import the data
datasetq2long = readtable("dataset_q2long.csv", opts);

% Clear temporary variables
clear opts
%%  Estimating transition parameters by ML

X_current = table2array(datasetq2long(:,"X_present"));
X_lag = table2array(datasetq2long(:, "X_past"));
Y_lag = table2array(datasetq2long(:, ["Y2", "Y3", "Y4"]));

ParamsLL_linear = @(Params) x_LL(X_current, X_lag, Y_lag,Params(1),Params(2:6));

%Optimization
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-10,'OptimalityTolerance',10^-10);
optimizers=fminunc(ParamsLL_linear, ones(6,1)',OptimOptions);


alpha = optimizers(2:6);
sigma = optimizers(1);

save("transition_pars", "alpha", "sigma")
%%
% ML estimation
clearvars
load('dataassign22.mat')
load("transition_pars.mat")
 
%V - remember to set subset X1t correctly
tic
x = V(1, ... %time
    2, ... %state
    X1, ... %X invariant
    X1t(:,1), ... % time variant regressor
    ones(5,1), ... %Gamma 1
    ones(5,1), ... %Gamma 2
    ones(5,1), ... %Delta
    0.5, ... %Trans cost
    alpha, ... %Trans X parameters 
    sigma, ... % Trans X parameter
    .95, ...%Discount rate
    1 );% epsilon draws
toc
%%
%Transition LL
function LL = x_LL(X, Xlag, Ylag, sigma, alpha)
N = length(X);
W = [ones(N, 1) Xlag Ylag];
LL = -1*(-(N/2).*log(sigma^2) - (1/(2*sigma^2))*sum((X - W*alpha').^2));
end

%% Conditional valuation function
%time/state/X1/X1t/gamma1/gamma2/delta/c/Alpha
function Vvalue=V(t,s,X1,X1t_col, gamma1,gamma2, delta,c,Alpha, sigma, beta, n_draws)
J = 5;
% one possible flow utility for each choice
u=zeros(length(X1), J);
u = repmat(u, 1, 1, n_draws);

% take repeated draws of epsilon's distribution, we will average over
% realizations
epsilon_draws = normrnd(0, sigma, length(X1), n_draws);
X1t_baseline = Alpha(1)*ones(size(X1t_col))+Alpha(2)*X1t_col;
X1taux=repmat(X1t_baseline, 1, n_draws)+epsilon_draws;

if t==9
    for j=2:5
        %Update X according to j and a random shock
        X1tUpdt=X1taux+(j >= 2 && j <= 4).*(Alpha(j).*ones(length(X1), n_draws));
        %Ut+1
        u(:, j, :)= X1tUpdt.*delta(j)+repmat(X1(:,1).*gamma1(j)+ X1(:,2).*gamma2(j)-c.*(s~=j), 1, n_draws);
    end
    % counterfactual utility in case choice j = 0 is made - no future
    % value term
    u(:, 1, :) =  X1tUpdt*delta(1)+ repmat(X1(:,1).*gamma1(1)+ X1(:,2).*gamma2(1)-c.*(s~=1), 1, n_draws);

    Vvalue=mean(log(sum(exp(u),2))+0.57721, 3);
else
    for j=2:5
        %Update X according to j and a random shock
        X1tUpdt=X1taux+(j >= 2 && j <= 4).*(Alpha(j).*ones(length(X1), n_draws));
        %Vt+1
        u(:, j, :)= X1tUpdt*delta(j)+ repmat(X1(:,1).*gamma1(j)+ X1(:,2).*gamma2(j)-c.*(s~=j), 1, n_draws) ...
            +beta*V(t+1,j,X1,mean(X1tUpdt, 2),gamma1,gamma2, delta,c,Alpha, sigma, beta, n_draws);
    end
    % counterfactual utility in case choice j = 0 is made - no future
    % value term
    u(:, 1, :) =  X1tUpdt*delta(1)+ repmat(X1(:,1).*gamma1(1)+ X1(:,2).*gamma2(1)-c.*(s~=1), 1, n_draws);

    Vvalue=mean(log(sum(exp(u),2))+0.57721, 3);
end
end