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

Alpha = optimizers(2:6);
sigma = optimizers(1);

% ML estimation
clearvars -except Alpha sigma 
load('dataassign22.mat')

%V
tic
x = V(9, ... %time
    3*ones(size(X1,1),1), ... %states
    X1, ... %X invariant
    X1t(:,10), ... %X variant
    ones(5,1), ... %Gamma 1
    ones(5,1), ... %Gamma 2
    ones(5,1), ... %Delta
    0, ... %Trans cost
    Alpha, ... %Trans X parameters 
    sigma, ...
    .95) ;%Discount rate
toc

%%
% Transition LL
function LL = x_LL(X, Xlag, Ylag, sigma, alpha)
N = length(X);
W = [ones(N, 1) Xlag Ylag];
LL = -1*(-(N/2).*log(sigma^2) - (1/(2*sigma^2))*sum((X - W*alpha').^2));
end

%% Conditional valuation function
%time/state/X1/X1t/gamma1/gamma2/delta/c/Alpha
function Vvalue=V(t,s,X1,X1t,gamma1,gamma2, delta,c,Alpha, sigma, beta)
    n=size(X1t,1);

    X1taux=Alpha(1)*ones(n,1)+Alpha(2)*X1t;
    %Update X according to j and a random shock
    X1tUpdt=repmat(X1taux(:,1),1,5)+repmat([0,0,Alpha(3:5)],n,1);

    if t==9
        u=X1tUpdt.*repmat(delta',n,1)+repmat(X1(:,1),1,5).*repmat(gamma1',n,1)+repmat(X1(:,2),1,5).*repmat(gamma2',n,1)-c*(repmat([1:5],n,1)~=s);
        Vvalue=log(sum(exp(u),2))+0.57721;
    else
        u=X1tUpdt.*repmat(delta',n,1)+repmat(X1(:,1),1,5).*repmat(gamma1',n,1)+repmat(X1(:,2),1,5).*repmat(gamma2',n,1)-c*(repmat([1:5],n,1)~=s);
        u=u+beta*([V(t+1,1,X1,X1tUpdt(1),gamma1,gamma2, delta,c,Alpha, sigma, beta), ...
            V(t+1,2,X1,X1tUpdt(2),gamma1,gamma2, delta,c,Alpha, sigma, beta), ...
            V(t+1,3,X1,X1tUpdt(3),gamma1,gamma2, delta,c,Alpha, sigma, beta), ...
            V(t+1,4,X1,X1tUpdt(4),gamma1,gamma2, delta,c,Alpha, sigma, beta), ...
            V(t+1,5,X1,X1tUpdt(5),gamma1,gamma2, delta,c,Alpha, sigma, beta)]);
        
        Vvalue=log(sum(exp(u),2))+0.57721;
    end
end


