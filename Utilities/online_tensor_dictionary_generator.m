function [X_observed,X_true,Factors,Core] = online_tensor_dictionary_generator(time_frame,tensor_dim,tensor_rank,fac_noise,outlier_fac,outlier_den,epsilon)
%  Author     : Le Trung Thanh
%  Affiliation: University of Orleans, France
%  Contact    : thanhle88.tbt@gmail.com 
%  Reference 
%  L.T. Thanh, T.T. Duy, K. Aded-Meraim, N.L. Trung, A. Hafiane 
%  "Robust Online Tucker Dictionary Learning From Multidimensional Data Streams"
%  In Proc. 14th APSIPA ASC, 2022.

N = length(tensor_dim);
T = time_frame;

Factors      = cell(T,N);
Core         = cell(T,1);
X_observed   = cell(T,1);
X_true       = cell(T,1);

sparsity = 0.3; % sparsity on tensor-cores;
%% t = 1;
Omega_t   = rand(tensor_rank);
Omega_t   = 1.*(Omega_t <= sparsity);
Core_1 = randn(tensor_rank);
Core_1 = tensor(Omega_t .* Core_1);

for n = 1 : N
    Factors{1,n} = randn(tensor_dim(n),tensor_rank(n));
end

X1          = ttm(Core_1,Factors(1,:),[1:N]);
X_true{1,1} = tensor(X1.data);
X_observed{1,1} = tensor(X1.data);
Core{1,1}   = Core_1;

%% t > 1
for t = 2 : T
    
    % Core tensor
    Omega_t   = rand(tensor_rank);
    Omega_t   = 1.*(Omega_t <= sparsity);
    RAND = randn(tensor_rank);
    RAND = Omega_t .* RAND;
    Core_t  = tensor(RAND);
    Core{t,1} = Core_t;
    % Factor
    if epsilon(t) == 1 % Abrupt changes
        for n = 1 : N
            Factors{t,n} = randn(tensor_dim(n),tensor_rank(n));
        end
    else
        for n = 1 : N
            Factors{t,n} = Factors{t-1,n} + epsilon(t)*randn(tensor_dim(n),tensor_rank(n));
            
        end
    end
    
    % Data
    data_true   = ttm(Core_t,Factors(t,:),[1:N]);
    
    Noise_Gauss = fac_noise*randn(tensor_dim);
    
    Outlier     = zeros(tensor_dim);
    NN          = prod(tensor_dim(1:end));
    p           = randperm(NN);
    L           = round(outlier_den*NN);
    Outlier(p(1:L))  = outlier_fac * rand(L,1);
    
    X_true{t,1}     = tensor(data_true);
    X_observed{t,1} = tensor(data_true + Noise_Gauss + Outlier);
    
end

end
