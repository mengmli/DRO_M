
for i=1:N_r % for each prescribed radius
    r=r_range(i);
    fprintf('order r %d ',i);
%     disp(r);
    n_disappt=0;
    for n=1:n_exper % run n_exper independent experiments
        q_T=naive_est_alpha(k,d,T,xi(:,:,n));
        cost_fin_iid(n)=10^6;
        for row=1:length(x_feasible(:,1))
            x_cur=x_feasible(row,:)'; %fix one decision
            cost_fin1 = w*cost_noM(a,k,x_cur,q_T,r,d);
            if cost_fin1<cost_fin_iid(n) %compare if it is the best decision so far
                cost_fin_iid(n)=cost_fin1;
                x=x_cur;
            end
        end
        cost_out_iid(n) = -(a.*x)'*alpha_real*w';
        
        if cost_out_iid(n)>cost_fin_iid(n)
            n_disappt=n_disappt+1;
        end
    end
    reliability_iid(i)=1-n_disappt/n_exper;

    iid_perf(i)=mean(cost_out_iid); 
    iid_perf_lower(i)=iid_perf(i)-2*std(cost_out_iid);
    iid_perf_upper(i)=iid_perf(i)+2*std(cost_out_iid);
end

save('T_500_simul.mat')

% figure(1)
% hold on;
% hmeaniid=plot(r_range, reliability_iid, 'LineWidth',2);
% hmeaniid.Color='r';
% hmean_m=plot(r_range, reliability,'LineWidth',2);
% hmean_m.Color='b';
% set(gca,'XScale','log')

% hold off;

figure(2)
hold on;
%markov
x3 = [r_range, fliplr(r_range)];
inBetween = [markov_perf_lower, fliplr(markov_perf_upper)];
h2=fill(x3, inBetween, 'b','Edgecolor', 'none');
set(h2,'FaceAlpha',0.2)
hmeanout=plot(r_range,markov_perf, 'b', 'LineWidth', 2);

%iid
x4 = [r_range, fliplr(r_range)];
inBetween2 = [iid_perf_lower, fliplr(iid_perf_upper)];
h3=fill(x4, inBetween2, 'r','Edgecolor', 'none');
set(h3,'FaceAlpha',0.2)
hmeanout_iid=plot(r_range,iid_perf, 'r', 'LineWidth', 2);

real=plot(r_range,cost_real+0*r_range, 'g', 'LineWidth', 2);
xlabel('r')
ylabel('cost')
% legend([hline],{'True cost'})
set(gca,'XScale','log')
hold off;