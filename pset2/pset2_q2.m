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

%Test path
XPath(zeros(5,1),ones(5,3),zeros(5,3),Alpha)


%% Transition LL
function LL = x_LL(X, Xlag, Ylag, sigma, alpha)
N = length(X);
W = [ones(N, 1) Xlag Ylag];
LL = -1*(-(N/2).*log(sigma^2) - (1/(2*sigma^2))*sum((W*alpha').^2));
end

%% Get the path of X given an initial value X, a sequence of future choices SC and a sequence of shocks Eps and transition parameters Alpha
% This is a function of Eps too because the plan is to use this function to evaluate integrals
%Note SC and Eps have the same length but SC starts at time t and Eps at
%t+1
function XPath=XPath(X,SC,Eps,Alpha)
    XPath=zeros(size(SC));
    XPath(:,1)=Alpha(1)*ones(size(X,1),1)+Alpha(2)*X+Alpha(SC(:,1)+2)'+Eps(:,1);
    for i=2:size(XPath,2)
        XPath(:,i)=Alpha(1)*ones(size(X,1),1)+Alpha(2)*XPath(:,i-1)+Alpha(SC(:,i-1)+2)'+Eps(:,i);
    end
end

