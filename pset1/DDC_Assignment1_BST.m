clear 
%% Load data
clear
load('dataassign2.mat')

filename = 'DDC_Assignment1Tables.xlsx';

Y0=(Y==0); 
Y1 = (Y==1); 
Y2 = (Y==2);
Y3 = (Y==3);
Y4 = (Y==4);
%% BST with X1 and Z as regressors
% We subtract Z from baseline level (1), and set Beta0 to be 0, implying
% baseline utility is 0.
Z=Z-1;

ParamsBST = @(Params) -bstLL(Y0,Y1,Y2,Y3, Y4, X1,Z, [0,0]' ,Params(1:2), Params(3:4), Params(5:6), Params(7:8), Params(9), Params(10), Params(11));

%Optimization - 
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-10,'OptimalityTolerance',10^-10);
S1_bst=fminunc(ParamsBST,0.2*ones(11,1),OptimOptions);
S2_bst=fminunc(ParamsBST,0.8*ones(11,1),OptimOptions);

Solutions=[S1_bst,S2_bst];

writematrix(Solutions,filename,'Sheet',1,'Range','B3')

%% BST likelihood

function bstLL = bstLL(Y0,Y1,Y2,Y3, Y4, X1,Z,Beta0,Beta1,Beta2, Beta3, Beta4,Gamma, Rho1, Rho2)

% to ease on notation, define mean utility for each alternative first

u_0 = X1*Beta0+ Z(:, 1)*Gamma;
u_1 = X1*Beta1+ Z(:, 2)*Gamma;
u_2 = X1*Beta2+ Z(:, 3)*Gamma;
u_3 = X1*Beta3+ Z(:, 4)*Gamma;
u_4 = X1*Beta4+ Z(:, 5)*Gamma;


a_1 = (1-Rho1)/(2 - Rho1 - Rho2);
a_2 = (1-Rho2)/(2 - Rho1 - Rho2);

G = a_1.*((exp(u_1./Rho1) + exp(u_2./Rho1)).^Rho1 + (exp(u_3./Rho1) + exp(u_4./Rho1)).^Rho1) ...
    + a_2.*((exp(u_1./Rho2) + exp(u_3./Rho2)).^Rho2 + (exp(u_2./Rho2) + exp(u_4./Rho2)).^Rho2) ...
    + exp(u_0);

Prob1 = (a_1 .* exp(u_1./Rho1).*(exp(u_1./Rho1) + exp(u_2./Rho1)).^(Rho1 - 1) ...
    + a_2.* exp(u_1./Rho2).*(exp(u_1./Rho2) + exp(u_3./Rho2)).^(Rho2 - 1))./G;

Prob2 = (a_1 .* exp(u_2./Rho1).*(exp(u_1./Rho1) + exp(u_2./Rho1)).^(Rho1 - 1) ...
    + a_2.* exp(u_2./Rho2).*(exp(u_2./Rho2) + exp(u_4./Rho2)).^(Rho2 - 1))./G;

Prob3 = (a_1 .* exp(u_3./Rho1).*(exp(u_3./Rho1) + exp(u_4./Rho1)).^(Rho1 - 1) ...
    + a_2.* exp(u_3./Rho2).*(exp(u_1./Rho2) + exp(u_3./Rho2)).^(Rho2 - 1))./G;

Prob4 = (a_1 .* exp(u_4./Rho1).*(exp(u_3./Rho1) + exp(u_4./Rho1)).^(Rho1 - 1) ...
    + a_2.* exp(u_4./Rho2).*(exp(u_2./Rho2) + exp(u_4./Rho2)).^(Rho2 - 1))./G;

Prob0 = 1 - Prob1 - Prob2 - Prob3 - Prob4;


bstLL = (Y0.')*log(Prob0) + (Y1.')*log(Prob1) + (Y2.')*log(Prob2) ...
    + (Y3.')*log(Prob3) + (Y4.')*log(Prob4);

end