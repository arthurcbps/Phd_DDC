%% Load data
clear 

load('dataassign1.mat')

%Since only differences in utility matter, substracting a constant from Z
%does not affect the model. This transformation sets utility from the first
%option to zero
Z=Z-1;

%Choice variables
Y0=(Y==0); Y1=(Y==1); Y2=(Y==2);

%File to store results
filename = 'DDC_Assignment1Tables.xlsx';

%% Multinomial logit with X1 and Z as regressors

%Multinomial LL as function of parameters only (Utility from the first option normalized to 0)
ParamsMLL = @(Params) -MLL(Y0,Y1,Y2,X1,X2,Z,[0,0]',0,Params(1:2),0,Params(3:4),0,Params(5));

%Optimization
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-10,'OptimalityTolerance',10^-10);

S=fminunc(ParamsMLL,-ones(5,1),OptimOptions);

writematrix(S,filename,'Sheet',1,'Range','B3')

%% Multinomial logit with X1, X2 and Z as regressors
%Multinomial LL as function of parameters only (Beta0 vector is normalized to 0)
ParamsMLL = @(Params) -MLL(Y0,Y1,Y2,X1,X2,Z,[0,0]',0,Params(1:2),Params(3),Params(4:5),Params(6),Params(7));

%Optimization
OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-10,'OptimalityTolerance',10^-10);
S=fminunc(ParamsMLL,-ones(7,1),OptimOptions);

writematrix(S,filename,'Sheet',1,'Range','E3')

%%  Log likelihood functions
function MLL = MLL(Y0,Y1,Y2,X1,X2,Z,Beta0_1,Beta0_2,Beta1_1,Beta1_2,Beta2_1,Beta2_2,Gamma)
    MLL= (Y0.')*(X1*Beta0_1+X2*Beta0_2+Z(:,1)*Gamma)...
        +(Y1.')*(X1*Beta1_1+X2*Beta1_2+Z(:,2)*Gamma)...
        +(Y2.')*(X1*Beta2_1+X2*Beta2_2+Z(:,3)*Gamma)...
        -Y0.'*log(exp(X1*Beta0_1+X2*Beta0_2+Z(:,1)*Gamma)+exp(X1*Beta1_1+X2*Beta1_2+Z(:,2)*Gamma)+exp(X1*Beta2_1+X2*Beta2_2+Z(:,3)*Gamma))...
        -Y1.'*log(exp(X1*Beta0_1+X2*Beta0_2+Z(:,1)*Gamma)+exp(X1*Beta1_1+X2*Beta1_2+Z(:,2)*Gamma)+exp(X1*Beta2_1+X2*Beta2_2+Z(:,3)*Gamma))...
        -Y2.'*log(exp(X1*Beta0_1+X2*Beta0_2+Z(:,1)*Gamma)+exp(X1*Beta1_1+X2*Beta1_2+Z(:,2)*Gamma)+exp(X1*Beta2_1+X2*Beta2_2+Z(:,3)*Gamma));
end

