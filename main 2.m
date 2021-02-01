clear all;
close all;
% dbstop if error
system('caffeinate &');
rng(400);
warning('off');
tic;
% reliability plot
d=5; % how many assets
m=5; % cap for values of asset
T=50; % length of each xi^(i)
N_sample=1; 

% p_i=[0.86 0.74 0.43 0.32 0.2];
% q_i=[0.12 0.5 0.2 0.3 0.4];
P =  rand(d,m);
B = 4*rand(d,1);
w=[ 0.2 0.4 0.1 0.1 0.2];
prime=rand(1,m);

%%
Ta = readtable('segmentation-data.csv');
Ta = Ta(:,[4 5 6]);
Ta.Income = (Ta.Income - mean(Ta.Income))/std(Ta.Income);

Ta=table2array(Ta);

[idx,C]=kmeans(Ta,d);

Ta = readtable('purchase data.csv');

Ta= Ta(Ta.Incidence==1,[1 4]);

Ta.ID=Ta.ID-200000000;
Ta=table2array(Ta);
for i=1:700
    Ta(Ta(:,1)==i)=idx(i);
end
collapsed=[];
test=[];
for i=1:d
    Ta_i=Ta(Ta(:,1)==i,2);
    collapsed(i,:)=Ta_i(1:T);
    test(i,:)=Ta_i((501):700);
end

save('xi.mat','collapsed');
matObj = matfile('xi.mat','Writable',true);

[nrows, ~] = size(matObj, 'collapsed');
xi=zeros(d,T,2);
xi(:,:,1)=matObj.collapsed;

%%
[true_x,alpha_real,cost_real] = test_real(w,prime,d,m,200,P,B,test);


epsilon=0.01;
iter=10;

cost_fin=zeros(1,N_sample); 
cost_out=zeros(1,N_sample); 
% 
cost_fin_iid=zeros(1,N_sample);
cost_out_iid=zeros(1,N_sample);
% 
cost_fin_saa=zeros(1,N_sample);
cost_out_saa=zeros(1,N_sample);

%%%%
xr=logspace(-3,0,60); %range r
[~,N_r]=size(xr);
y=zeros(1,N_r);
y1=zeros(1,N_r);
y2=zeros(1,N_r);
reliability=zeros(1,N_r);
reliability_iid=zeros(1,N_r);

z=zeros(1,N_r);
z1=zeros(1,N_r);
z2=zeros(1,N_r);

w0=zeros(1,N_r);
w1=zeros(1,N_r);
w2=zeros(1,N_r);

options = optimoptions('fmincon','Display', 'off');
% options = optimoptions(options,'MaxFunctionEvaluations', 300000);
% options = optimoptions(options,'MaxIterations', 300000);
% options = optimoptions(options,'OptimalityTolerance', 0.0001);
% options = optimoptions(options,'FunctionTolerance', 0.0001);
% options = optimoptions(options,'StepTolerance', 0.0001);
xrange=dec2bin(0:1:2^m-1)-'0';   
i=1;
for r=xr
    fprintf('order r %d ',i);
%     disp(r);
    n_disappt=0;
  
for nsample=1:N_sample
    fprintf('exp order %d ',nsample);
    [alpha0,q]=est_alpha_from_xi(d,m,T,xi(:,:,nsample));
% if r>=0.1
%     options=optimoptions(options,'MaxFunctionEvaluations', 3000);
%     options = optimoptions(options,'MaxIterations', 3000);  
%     end
    cost_fin(nsample)=10^6;
    for row=1:32
        xrow=xrange(row,:)';
        if P*xrow<=B
            cost_fin1 = -w*FW_main(d,xrow,prime,epsilon,r,iter,q,m,alpha0);
        else 
            cost_fin1 = 10^6;
        end
        if cost_fin1<cost_fin(nsample)
            cost_fin(nsample)=cost_fin1;
            x=xrow;
        end
    end
    cost_out(nsample) = -(prime'.*x)'*alpha_real*w';

    if cost_out(nsample)>cost_fin(nsample)
        n_disappt=n_disappt+1;
    end
end
reliability(i)=1-n_disappt/N_sample;

z(i)=mean(cost_out);
z1(i)=z(i)-2*std(cost_out);
z2(i)=z(i)+2*std(cost_out);
i=i+1;
end

i=1;
for r=xr
    n_disappt=0;
for nsample=1:N_sample
    alpha0=naive_est_alpha(d,m,T,xi(:,:,nsample));
    cost_fin_iid(nsample)=10^6;
    for row=1:32
        xrow=xrange(row,:)';
        if P*xrow<=B
            cost_fin1 = -w*cost_noM(d,xrow,prime,alpha0,r,m);
        else 
            cost_fin1 = 10^6;
        end
        if cost_fin1<cost_fin_iid(nsample)
            cost_fin_iid(nsample) =cost_fin1;
            x=xrow;
        end
    end
    cost_out_iid(nsample) = -(prime'.*x)'*alpha_real*w';
    if cost_out_iid(nsample)>cost_fin_iid(nsample)
        n_disappt=n_disappt+1;
    end
end
reliability_iid(i)=1-n_disappt/N_sample;


w0(i)=mean(cost_out_iid); 
w1(i)=w0(i)-2*std(cost_out_iid);
w2(i)=w0(i)+2*std(cost_out_iid);
i=i+1;
end

n_disappt=0;
for nsample=1:N_sample
    alpha0=naive_est_alpha(d,m,T,xi(:,:,nsample));
    cost_fin_saa(nsample)=10^6;
    for row=1:32
        xrow=xrange(row,:)';
        if P*xrow<=B
            cost_fin1 = -(prime'.*xrow)'*alpha0'*w';
        else 
            cost_fin1 = 10^6;
        end
        if cost_fin1<cost_fin_saa(nsample)
            cost_fin_saa(nsample)=cost_fin1;
            x=xrow;
        end
    end
    cost_out_saa(nsample) = -(prime'.*x)'*alpha_real*w';
    if cost_out_saa(nsample)>cost_fin_saa(nsample)
        n_disappt=n_disappt+1;
    end
    
end
reliability_SAA=1-n_disappt/N_sample+0*xr;
saa_out=mean(cost_out_saa)+0*xr;
saa_2=mean(cost_out_saa)-2*std(cost_out_saa)+0*xr;
saa_3=mean(cost_out_saa)+2*std(cost_out_saa)+0*xr;


figure(1)
hold on;
hmeansaa=plot(xr, reliability_SAA, 'LineWidth',2);
hmeansaa.Color='y';
hmeaniid=plot(xr, reliability_iid, 'LineWidth',2);
hmeaniid.Color='r';
hmean_m=plot(xr, reliability,'LineWidth',2);
hmean_m.Color='b';
set(gca,'XScale','log')

hold off;

figure(2)
hold on;
%markov
x3 = [xr, fliplr(xr)];
inBetween = [z1, fliplr(z2)];
h2=fill(x3, inBetween, 'b','Edgecolor', 'none');
set(h2,'FaceAlpha',0.2)
hmeanout=plot(xr,z, 'b', 'LineWidth', 2);

%iid
x4 = [xr, fliplr(xr)];
inBetween2 = [w1, fliplr(w2)];
h3=fill(x4, inBetween2, 'r','Edgecolor', 'none');
set(h3,'FaceAlpha',0.2)
hmeanout_iid=plot(xr,w0, 'r', 'LineWidth', 2);

real=plot(xr,cost_real+0*xr, 'g', 'LineWidth', 2);
xlabel('r')
ylabel('cost')
% legend([hline],{'True cost'})
set(gca,'XScale','log')
hold off;

figure(3)
hold on;
%markov
x3 = [xr, fliplr(xr)];
inBetween = [z1, fliplr(z2)];
h2=fill(x3, inBetween, 'b','Edgecolor', 'none');
set(h2,'FaceAlpha',0.2)
hmeanout=plot(xr,z, 'b', 'LineWidth', 2);

%saa
x4 = [xr, fliplr(xr)];
inBetween2 = [saa_2, fliplr(saa_3)];
h3=fill(x4, inBetween2, 'y','Edgecolor', 'none');
set(h3,'FaceAlpha',0.2)
hmeanout_saa=plot(xr,saa_out, 'y', 'LineWidth', 2);

real=plot(xr,cost_real+0*xr, 'g', 'LineWidth', 2);
xlabel('r')
ylabel('cost')
% legend([hline],{'True cost'})
set(gca,'XScale','log')


% FW_main.m
% grad_pi_k.m
% Grad_Psi.m
% linear_sub.m
% main.m
% minor.m
% naive_est_alpha.m
% Psi.m
% sample.m
% so_minor.m
% test_real.m

save('simulation_data')
t=toc;
etime=t./60
% matlab2tikz('T200.tex');

