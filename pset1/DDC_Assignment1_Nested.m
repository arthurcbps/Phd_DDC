%% Load data
clear 

load('dataassign1.mat')

%Choice variables
Y0=(Y==0); Y1=(Y==1); Y2=(Y==2);

% normalization wrt to base alternative
Z = Z-1;

%File to store results
filename = 'DDC_Assignment1Tables_nested1.xlsx';

%% Nested logit with X1 and Z as regressors
%Nested logit LL as function of parameters only (we normalize beta_0 to be a null vector)
ParamsNestedLL = @(Params) -NestedLL(Y0,Y1,Y2,X1,X2,Z,[0,0]',0,Params(1:2),0,Params(3:4),0,Params(5), Params(6));

%Optimization - results are robust to different initial conditions
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-10,'OptimalityTolerance',10^-10);
S1_Nested=fminunc(ParamsNestedLL, 0.2*ones(6,1),OptimOptions);
S2_Nested=fminunc(ParamsNestedLL,0.8*ones(6,1),OptimOptions);

writematrix(S1_Nested,filename,'Sheet',1)


%% Nested logit with X1, X2 and Z as regressors
%Nested logit LL as function of parameters only (we normalize beta_0 to be a null vector)
ParamsNestedLL = @(Params) -NestedLL(Y0,Y1,Y2,X1,X2,Z,[0,0]',0,Params(1:2),Params(3),Params(4:5),Params(6),Params(7), Params(8));

%Optimization
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-10,'OptimalityTolerance',10^-10);
S1_Nested2=fminunc(ParamsNestedLL,0.2*ones(8,1),OptimOptions);
S2_Nested2=fminunc(ParamsNestedLL,0.8*ones(8,1),OptimOptions);

Solutions_Nested2=[S1_Nested2,S2_Nested2];

filename = 'DDC_Assignment1Tables_nested2.xlsx';
writematrix(S1_Nested2,filename,'Sheet',1)

%% Nested logit likelihood

function NestedLL = NestedLL(Y0,Y1,Y2,X1,X2,Z,Beta0_1,Beta0_2,Beta1_1,Beta1_2,Beta2_1,Beta2_2,Gamma, Rho)

% to ease on notation, define mean utility for each alternative first

u_0 = X1*Beta0_1+X2*Beta0_2+Z(:,1)*Gamma;
u_1 = X1*Beta1_1+X2*Beta1_2+Z(:,2)*Gamma;
u_2 = X1*Beta2_1+X2*Beta2_2+Z(:,3)*Gamma;

G = (exp(u_1./Rho) + exp(u_2./Rho)).^Rho + exp(u_0);
Prob1 = exp(u_1./Rho).*(exp(u_1./Rho) + exp(u_2./Rho)).^(Rho-1)./G;
Prob2 = exp(u_2./Rho).*(exp(u_1./Rho) + exp(u_2./Rho)).^(Rho-1)./G;
Prob0 = 1 - Prob1 - Prob2;

NestedLL = (Y0.')*log(Prob0) + (Y1.')*log(Prob1) + (Y2.')*log(Prob2);
end