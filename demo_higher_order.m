clear;clc; close all;

run_path;

%% N-order streaming tensors
n_exp       = 2;
time_frame  = 500;                       % number of temporal slices
tensor_dim  = [10 10 10 10];             % dimension of temporal slices
tucker_rank = [3 3 3 3];
fac_noise   = 1e-3;
outlier_fac = 10;
time_varying = 1e-3;
epsilon      = time_varying*ones(time_frame,1);
epsilon(300) = 1;

outlier_den_vec = [0.1];

PER_SEP = zeros(length(outlier_den_vec),time_frame);

for jj = 1 : length(outlier_den_vec)
    outlier_den = outlier_den_vec(jj);
    
    PER = zeros(1,time_frame);
    for ii = 1 : n_exp
        fprintf('+ run (%d/%d)  \n',jj,ii)
        %% Generate streaming data
        [X_cell,X_true,Factor_True,Core_True] = online_tensor_dictionary_generator(time_frame,...
            tensor_dim,tucker_rank,fac_noise,outlier_fac,outlier_den,epsilon);
        
        %% Main Program
        OPTS.Factor_True = Factor_True;
        OPTS.Slide_True  = X_true;
        OPTS.Core_True   = Core_True;
        [~,PER_ii]  = ROTDL(X_cell,tucker_rank,OPTS);
        PER = PER + PER_ii(1,:);
    end
    PER = PER / n_exp;
    PER_SEP(jj,:) = PER;
end

%% Plot
makerSize = 14;
numbMarkers = 50;
LineWidth = 2;
set(0, 'defaultTextInterpreter', 'latex');
color   = get(groot,'DefaultAxesColorOrder');
red_o   = [1,0,0];
blue_o  = [0, 0, 1];
gree_o  = 'g'; %[0, 0.5, 0];
black_o = [0.25, 0.25, 0.25];

blue_n  = color(1,:);
oran_n  = color(2,:);
yell_n  = color(3,:);
viol_n  = color(4,:);
gree_n  = color(5,:);
lblu_n  = color(6,:);
brow_n  = color(7,:);
lbrow_n = [0.5350    0.580    0.2840];



fig = figure;
k = 1;
hold on;

d1 = semilogy(1:k:time_frame,PER_SEP(1,1:k:end),...
    'linestyle','-','color',blue_o,'LineWidth',LineWidth);
d11 = plot(1:50:time_frame,PER_SEP(1,1:50:end),...
 'marker','o','markersize',makerSize,...
   'linestyle','none','color',blue_o,'LineWidth',2);
d12 = semilogy(1:1,PER_SEP(1,1),...
    'marker','o','markersize',makerSize,...
    'linestyle','-','color',blue_o,'LineWidth',2);


% d2 = semilogy(1:k:time_frame,PER_SEP(2,1:k:end),...
%     'linestyle','-','color',red_o,'LineWidth',LineWidth);
% d21 = plot(1:50:time_frame,PER_SEP(2,1:50:end),...
%  'marker','d','markersize',makerSize,...
%    'linestyle','none','color',red_o,'LineWidth',2);
% d22 = semilogy(1:1,PER_SEP(2,1),...
%     'marker','d','markersize',makerSize,...
%     'linestyle','-','color',red_o,'LineWidth',2);
% 
% 
% 
% d3 = semilogy(1:k:time_frame,PER_SEP(3,1:k:end),...
%     'linestyle','-','color',black_o,'LineWidth',LineWidth);
% d31 = plot(1:50:time_frame,PER_SEP(3,1:50:end),...
%  'marker','s','markersize',makerSize,...
%    'linestyle','none','color',black_o,'LineWidth',2);
% d32 = semilogy(1:1,PER_SEP(3,1),...
%     'marker','s','markersize',makerSize,...
%     'linestyle','-','color',black_o,'LineWidth',2);
% 
% 
% lgd = legend([d12 d22 d32],'$\omega_{outlier} = 10\%$','$\omega_{outlier} = 30\%$','$\omega_{outlier} = 50\%$');
% lgd.FontSize = 28;
% set(lgd, 'Interpreter', 'latex', 'Color', [0.95, 0.95, 0.95]);

ylabel('SEP $(\mathcal{D}_{tr}, \mathcal{D}_{es})$','interpreter','latex','FontSize',13,'FontName','Times New Roman');
xlabel('Time Index','interpreter','latex','FontSize',13,'FontName','Times New Roman');

% 
h=gca;
set(gca, 'YScale', 'log');


yticks([1e-8 1e-5 1e-2  1e1 ])
yticklabels({'10^{-8}','10^{-5}','10^{-2}','10^{1}'})

set(h,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle','-','MinorGridLineStyle','-','FontName','Times New Roman');
set(h,'Xtick',0:100:time_frame,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle',':','MinorGridLineStyle','none',...
    'FontName','Times New Roman');
set(h,'FontSize', 30);
grid on;
box on;
axis([0 time_frame 1e-8 1e1])
set(fig, 'units', 'inches', 'position', [0.5 0.5 10 7]);

