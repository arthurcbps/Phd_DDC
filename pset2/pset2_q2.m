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

% ML estimation of transition parameters
clearvars -except Alpha sigma 
load('dataassign22.mat')

%Normal draws (1,-1 with equal probability is symmetric around 0 and has variance 1)
Eps=normrnd(0,sigma,5,1);

%ML estimation of the model
tic
%Optimization
Params_LL= @(Params) LL(LY1, Y, X1,X1t, Params(1:4),Params(5:8), Params(9:12),Params(13),Alpha, sigma, Params(14),Eps);
%Params_LL(ones(14,1))

OptimOptions = optimoptions(@fminunc,'Display','Iter','StepTolerance',10^-6,'OptimalityTolerance',10^-6);
S=fminunc(Params_LL,0.5*ones(14,1),OptimOptions);
toc
%% Transition LL
function LL = x_LL(X, Xlag, Ylag, sigma, alpha)
N = length(X);
W = [ones(N, 1) Xlag Ylag];
LL = -1*(-(N/2).*log(sigma^2) - (1/(2*sigma^2))*sum((X - W*alpha').^2));
end

%% Conditional valuation function
%time/state/X1/X1t/gamma1/gamma2/delta/c/Alpha
function Vvalue=V(t,s,X1,X1t,gamma1,gamma2, delta,c,Alpha, sigma, beta,Eps)
    
    %Get sizes
    n=size(X1t,1);
    draws=length(Eps);
    Vvalue=zeros(n,5);


    %Get immediate utilities
    u=zeros(n,5);    
    u(:,2:5)=repmat(X1t,1,4).*repmat(delta',n,1)+repmat(X1(:,1),1,4).*repmat(gamma1',n,1)+repmat(X1(:,2),1,4).*repmat(gamma2',n,1)-c*(repmat(1:4,n,1)~=s);
      
    if t==10
        
        Vvalue=u;
    
    else

        %Get immediate utilities
        u=zeros(n,5);
        u(:,2:5)=repmat(X1t,1,4).*repmat(delta',n,1)+repmat(X1(:,1),1,4).*repmat(gamma1',n,1)+repmat(X1(:,2),1,4).*repmat(gamma2',n,1)-c*(repmat(1:4,n,1)~=s);
        
        %Matrix to store results over simulations
        v=zeros(n*draws,5);

        %Update X according to j 
        X1taux=Alpha(1)*ones(n,1)+Alpha(2)*X1t;
        X1tUpdt=repmat(X1taux(:,1),1,4)+repmat([0,Alpha(3:5)],n,1);
        
        %Expand over simulated normal errors
        X1tUpdt=repelem(X1tUpdt,draws,1)+repelem(Eps,n,1);
 
        for j=2:5
            
            %Calculate and add expected optimal behavior
            v(:,j)=beta*(log(sum(exp(V(t+1,j-1,repelem(X1,draws,1),X1tUpdt(:,j-1),gamma1,gamma2, delta,c,Alpha, sigma, beta,Eps)),2))+0.57721);
            
            %Average over simulated normal errors
            Vvalue(:,j)=u(:,j)+sum(reshape(v(:,j),draws,[]),1)';

        end

    end
end


%% Log likelihood
function LL=LL(LY1, Y, X1,X1t, gamma1,gamma2, delta,c,Alpha, sigma, beta,Eps)
 
   Y=[LY1,Y];
   LL=0;

   for t=8:10
       
       CV=V(t,Y(Y(:,t)~=0,t),X1(Y(:,t)~=0,:),X1t(Y(:,t)~=0,t),gamma1,gamma2, delta,c,Alpha, sigma, beta,Eps);

       n=size(Y(Y(:,t)~=0,t),1);

       num=max(exp(CV).*(repmat((0:4)',1,n)==Y(Y(:,t)~=0,t+1)')',[],2);
       den=sum(exp(CV),2);       

       LL=LL-sum(log(num)-log(den));

   end

end

