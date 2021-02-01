function [x_true,alpha_real, val] = test_real(d,k,test,P,B,w,a,x_feasible)
    [alpha_real,~] = est_alpha_from_xi(k,d,length(test),test); %calculating stationary distr.

    xrange=x_feasible;
    val=10^6;
    for row=1:length(xrange(:,1))
        x=xrange(row,:)';
        val1 = -(a.*x)'*alpha_real*w';
        
        if val1<val
            val=val1;
            x_true=x;
        end
    end
    
    disp('true x');
    disp(x_true);
    end
    