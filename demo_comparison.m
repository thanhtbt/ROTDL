clear;clc; close all;

run_path;

%% N-order streaming tensors
n_exp       = 3;
time_frame  = 500;                       % number of temporal slices
tensor_dim  = [10 10 10];                 % dimension of temporal slices
tucker_rank = [3 3 3];
epsilon     = 1e-3*ones(time_frame,1);
epsilon(300)= 1;
outlier_den = 0.2;
outlier_fac = 10;

fac_noise = 1e-3;

%% 
Omega = cell(time_frame,1);
for ii = 1 : time_frame
    Omega{ii,1} = ones(tensor_dim);
end

PER_ROTDL = zeros(1,time_frame);
PER_DTA = zeros(1,time_frame);
PER_STA = zeros(1,time_frame);
PER_ATD = zeros(1,time_frame);

N = length(tensor_dim);
for ii = 1 : n_exp
    fprintf('+ run %d  \n',ii)
    %% Generate streaming data
    [X_cell,X_true,Factor_True,Core_True] = online_tensor_dictionary_generator(time_frame,...
        tensor_dim,tucker_rank,fac_noise,outlier_fac,outlier_den,epsilon);
    
    %% Main Program
    OPTS.Factor_True = Factor_True;
    OPTS.Slide_True  = X_true;
    OPTS.Core_True   = Core_True;
    [~,PER_ROTDL_ii]  = ROTDL(X_cell,tucker_rank,OPTS);
    PER_ROTDL = PER_ROTDL + PER_ROTDL_ii(1,:);
%     
    [~,PER_DTA_ii]  = DTA_Tracking(X_cell,tucker_rank,OPTS);
    PER_DTA = PER_DTA + PER_DTA_ii(1,:);
    
    [~,PER_STA_ii]  = STA_Tracking(X_cell,tucker_rank,OPTS);
    PER_STA = PER_STA + PER_STA_ii(1,:);
    
    rank_ATD = [tucker_rank tucker_rank(N)];
    
    [~,PER_ATD_ii]  = ATD(X_cell,Omega,rank_ATD,OPTS);
    PER_ATD = PER_ATD + PER_ATD_ii(1,:);
    
end
PER_ROTDL = PER_ROTDL / n_exp;
PER_DTA   = PER_DTA / n_exp;
PER_STA   = PER_STA / n_exp;
PER_ATD   = PER_ATD / n_exp;


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

d2 = semilogy(1:k:time_frame,PER_DTA(1,1:k:end),...
    'linestyle','-','color',black_o,'LineWidth',LineWidth);
d21 = plot(1:50:time_frame,PER_DTA(1,1:50:end),...
    'marker','d','markersize',makerSize,...
    'linestyle','none','color',black_o,'LineWidth',2);
d22 = semilogy(1:1,PER_DTA(1,1),...
    'marker','d','markersize',makerSize,...
    'linestyle','-','color',black_o,'LineWidth',2);


d3 = semilogy(1:k:time_frame,PER_STA(1,1:k:end),...
    'linestyle','-','color',gree_o,'LineWidth',LineWidth);
d31 = plot(1:50:time_frame,PER_STA(1,1:50:end),...
    'marker','p','markersize',makerSize,...
    'linestyle','none','color',gree_o,'LineWidth',2);
d32 = semilogy(1:1,PER_STA(1,1),...
    'marker','p','markersize',makerSize,...
    'linestyle','-','color',gree_o,'LineWidth',2);

d4 = semilogy(1:k:time_frame,PER_ATD(1,1:k:end),...
    'linestyle','-','color',blue_o,'LineWidth',LineWidth);
d41 = plot(1:50:time_frame,PER_ATD(1,1:50:end),...
    'marker','h','markersize',makerSize,...
    'linestyle','none','color',blue_o,'LineWidth',2);
d42 = semilogy(1:1,PER_ATD(1,1),...
    'marker','h','markersize',makerSize,...
    'linestyle','-','color',blue_o,'LineWidth',2);


d1 = semilogy(1:k:time_frame,PER_ROTDL(1,1:k:end),...
    'linestyle','-','color',red_o,'LineWidth',LineWidth);
d11 = plot(1:50:time_frame,PER_ROTDL(1,1:50:end),...
    'marker','o','markersize',makerSize,...
    'linestyle','none','color',red_o,'LineWidth',2);
d12 = semilogy(1:1,PER_ROTDL(1,1),...
    'marker','o','markersize',makerSize,...
    'linestyle','-','color',red_o,'LineWidth',2);


lgd = legend([ d22 d32 d42 d12],'\texttt{DTA}','\texttt{STA}','\texttt{ATD}','\texttt{ROTDL}');
lgd.FontSize = 28;
set(lgd, 'Interpreter', 'latex', 'Color', [0.95, 0.95, 0.95]);

ylabel('SEP $(\mathcal{D}_{tr}, \mathcal{D}_{es})$','interpreter','latex','FontSize',13,'FontName','Times New Roman');
xlabel('Time Index','interpreter','latex','FontSize',13,'FontName','Times New Roman');

%
h=gca;
set(gca, 'YScale', 'log');


yticks([1e-5 1e-3 1e-1  1e1])
yticklabels({'10^{-5}','10^{-3}','10^{-1}','10^{1}'})

set(h,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle','-','MinorGridLineStyle','-','FontName','Times New Roman');
set(h,'Xtick',0:100:time_frame,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle',':','MinorGridLineStyle','none',...
    'FontName','Times New Roman');
set(h,'FontSize', 30);
grid on;
box on;
axis([0 time_frame 1e-5 1e1])
set(fig, 'units', 'inches', 'position', [0.5 0.5 10 7]);

