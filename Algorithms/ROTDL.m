function [Factor_Es,PER] = ROTDL(X_cell,tucker_rank,OPTS)
%  Author     : Le Trung Thanh
%  Affiliation: University of Orleans, France
%  Contact    : thanhle88.tbt@gmail.com 
%  Reference 
%  L.T. Thanh, T.T. Duy, K. Aded-Meraim, N.L. Trung, A. Hafiane 
%  "Robust Online Tucker Dictionary Learning From Multidimensional Data Streams"
%  In Proc. 14th APSIPA ASC, 2022.

%% 
if nargin <= 2
    flag = 0; % without performance estimation part
    lambda  = 0.5;
    Factors = []; PER = [];
else
    flag = 1; % with performance estimation part
    if isfield(OPTS,'Factor_True')
        Factor_True  = OPTS.Factor_True;
        flag_factor = 1;
    else
        flag_factor = 0;
    end
    if isfield(OPTS,'Slide_True')
        Slide_True = OPTS.Slide_True;
        flag_slide = 1;
    else
        flag_slide = 0;
    end
    if isfield(OPTS,'lambda')
         lambda = OPTS.lambda;
    else lambda = 0.5;
    end
    
end

T          = length(X_cell);
tensor_dim = size(X_cell{1,1});
N          = length(tensor_dim);
PER        = zeros(2,T);

%% Random Initialization
Core      = tensor(randn(tucker_rank));
Factor_Es = cell(1,N);
S         = cell(1,N);
for ii = 1 : N
    Factor_Es{1,ii} = randn(tensor_dim(ii),tucker_rank(ii));
    S{1,ii}         = 100*eye(tucker_rank(ii));
end

%% Main Program
for t = 1 : T
    
    X_t = X_cell{t,1};    
    x_t = X_t(:);   
    
    % Tensor coding
    H_t   = Factor_Es{1,N};
    for n = N-1 : -1 : 1
        H_t = kron(H_t,(Factor_Es{1,n}));
    end
    [ot,gt]     = tensor_coding(x_t,H_t,[]);
    idx_ot      = find(ot);
    Omega_ot_ll = ones(size(H_t,1),1);
    Omega_ot_ll(idx_ot) = 0;
    Omega_ot    = Omega_ot_ll;
    Omega_Ot    = tensor(reshape(Omega_ot,tensor_dim));
    
    er_t  = x_t - H_t*gt - ot;
    ER_t  = tensor(reshape(er_t,tensor_dim));
    ER_t  = Omega_Ot .* ER_t;
    G_t   = tensor(reshape(gt,tucker_rank));
    
    % Dictionary update
    for n = 1 : N
        if n == 1 
            W_n  = ttm(G_t,Factor_Es,[2:N]);
        else
            W_n  = ttm(G_t,Factor_Es,[1:n-1 n+1:N]);
        end
        W_n = ten2mat(W_n,[n]);
        ER_unfolding_n = ten2mat(ER_t,[n]);
        S{1,n} = W_n * W_n' + lambda*S{1,n};
        V_n    = (S{1,n}) \ W_n;
        Factor_Es{1,n} = Factor_Es{1,n} + ER_unfolding_n * V_n';
    end
    
    % Performance evaluation
    if flag == 1
        per_t = 0;
        for n = 1 : N
            per_t  =  per_t + sub_est_per(Factor_Es{1,n},Factor_True{t,n},'SEP');
        end
        PER(1,t) = per_t / N;
        
    else
    end
end

end

function [o,g] = tensor_coding(x,H,OPTS)
%% This ADMM solver is faster than the original LASSO

if isfield(OPTS,'RHO'),
    rho = OPTS.RHO;
else
    rho = 1;
end
if isfield(OPTS,'MAX_ITER'),
    MAX_ITER = OPTS.MAX_ITER;
else
    MAX_ITER = 50;
end

ABSTOL   = 1e-4;
RELTOL   = 1e-2;
%% ADMM 
[n,r] = size(H);
q = randn(n,1);
o = randn(n,1);
z = zeros(n,1);

g_re  = H' * x;
[L U] = factorize(H,1);
for k = 1 : MAX_ITER
  
    g_tem = g_re + H' * (q - o);
    g = U \ (L \ g_tem);
    x_re = H*g;
    tmp = x_re - x + o;
    q = 1/(1+1/rho)*tmp + 1/(1+rho)*shrinkage(tmp, 1+1/rho);

    u = 1/(rho+1)  * (x - x_re  - rho*(o - z));
    o_old = o;
    o = shrinkage(u + z, 1/rho);
    z = z + (u - o);
    
    s_norm(k)  = norm(-rho*(o - o_old));
    eps_dual(k)= sqrt(r)*ABSTOL + RELTOL*norm(rho*z);
    if s_norm(k) < eps_dual(k)
        break;
    end
end
o(abs(o)< 1) = 0;
end

function y = shrinkage(a, alpha)
y = max(0, a-alpha) - max(0, -a-alpha);
end




