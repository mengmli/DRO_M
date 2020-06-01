% function [theta_T,val0,q_all] = alpha_estimate(d,m,T,p_i,q_i)
% clear all %used to generate usable xi
d=5; % how many assets
m=10; % cap for values of asset
T=50; % length of each xi^(i)
% for n=1:10^7
xi=zeros(d,T);
%MC simul
p_i=[0.86 0.74 0.43 0.32 0.88];
q_i=[0.12 0.05 0.02 0.03 0.12];
p=p_i;q=q_i;
for i=1:d
xi(i,1)=0; %initial value for each asset
for t=2:T
    po=rand;
    if xi(i,t-1)==0
        if po<p(i)
            xi(i,t)=xi(i,t-1)+1;
        else
            xi(i,t)=xi(i,t-1);
        end
    elseif xi(i,t-1)==m-1
        if po>(1-q(i))
            xi(i,t)=xi(i,t-1)-1;
        else
            xi(i,t)=xi(i,t-1);
        end
    else
    if po<p(i)
        xi(i,t)=xi(i,t-1)+1;
    elseif po>(1-q(i))
        xi(i,t)=xi(i,t-1)-1;
    else
        xi(i,t)=xi(i,t-1);
    end
    end
end
end
% disp(xi);

%estimator theta_T
theta_T=zeros(m,m,d);
for i=1:d 
theta_q=zeros(m);
for k=1:m
    for j=1:m
        sum1=0;
        for t=1:(T-1)
            sum1=sum1+(xi(i,t)==k-1)*(xi(i,t+1)==j-1);
        end
        if sum1==0
            theta_q(k,j)=10^(-32);
        else
        theta_q(k,j)=sum1/(T-1);
        end
    end
end
theta_T(:,:,i)=theta_q;
end

%estimate transition matrix q_ij
q_all=zeros(m,m,d);
for i=1:d 
q=zeros(m);
for k=1:m
    for j=1:m
        if (sum(theta_T(k,1:m,i))==0)
            error('division by zero computing q'); 
%             disp(theta_T(k,1:m,i)); disp(theta_T);return
%             q(i,j)=theta_q(i,j)/0.00001;
        else
            q(k,j)=theta_T(k,j,i)/sum(theta_T(k,1:m,i));
        end
    end
end
q_all(:,:,i)=q;
end
xi=reshape(xi.',1,[]); %flatten xi matrix
% save('xi_new.mat','xi');
matObj = matfile('xi_new.mat','Writable',true);
% Find Size
[nrows, ~] = size(matObj, 'xi');
matObj.xi(nrows+1,:) = xi;

% disp(q);

%estimator p(i)
% P_T=zeros(m,d);
% for i=1:d
%     sum1=0;
%     for j=1:m
%         sum1=sum1+(xi(i,t)==k-1);
%     end
% end

% end