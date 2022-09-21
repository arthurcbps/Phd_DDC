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
Alpha=fminunc(ParamsLL_linear, ones(6,1)',OptimOptions);

%% ML estimation
clearvars -except Alpha
load('dataassign22.mat')

%V
tic
V(7, ... %time
    3*ones(size(X1,1),1), ... %states
    X1, ... %X invariant
    X1t(:,10), ... %X variant
    zeros(5,1), ... %Gamma 1
    zeros(5,1), ... %Gamma 2
    ones(5,1), ... %Delta
    0, ... %Trans cost
    Alpha, ... %Trans X parameters 
    .95) %Discount rate
toc

%% Transition LL
function LL = x_LL(X, Xlag, Ylag, sigma, alpha)
N = length(X);
W = [ones(N, 1) Xlag Ylag];
LL = -1*(-(N/2).*log(sigma^2) - (1/(2*sigma^2))*sum((W*alpha').^2));
end

%% Conditional valuation function
%time/state/X1/X1t/gamma1/gamma2/delta/c/Alpha
function Vvalue=V(t,s,X1,X1t,gamma1,gamma2, delta,c,Alpha,beta)
    u=zeros(size(X1));
    X1taux=Alpha(1)*ones(size(X1t,1),1)+Alpha(2)*X1t+normrnd(0,1,size(X1,1),1);
    if t==9
        for j=1:5
            %Update X according to j and a random shock
            X1tUpdt=X1taux+Alpha(j+1)*ones(size(X1t));
            %Ut+1
            u(:, j)= X1tUpdt.*delta(j)+X1(:,1).*gamma1(j)+ X1(:,2).*gamma2(j)-c.*(s~=j);
        end
        Vvalue=log(sum(exp(u),2))+0.57721;
    else
        for j=1:5
            %Update X according to j and a random shock
            X1tUpdt=X1taux+Alpha(j+1)*ones(size(X1t));
            %Vt+1
            u(:, j)= X1tUpdt.*delta(j)+ X1(:,1).*gamma1(j)+ X1(:,2).*gamma2(j)-c.*(s~=j)+beta*V(t+1,j,X1,X1tUpdt,gamma1,gamma2, delta,c,Alpha,beta);
        end
        Vvalue=log(sum(exp(u),2))+0.57721;
    end
end