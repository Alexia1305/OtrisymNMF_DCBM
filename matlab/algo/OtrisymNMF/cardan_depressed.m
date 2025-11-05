function [x_opt,f_opt] = cardan_depressed(c3,c1,c0,default_x)
% find x that min c3/4*x^4+c1/2*x^2+c0*x and x>0
% default value otherwise
% solve by cardano formula 

    x_opt=default_x;
    f_opt=(c3/4)*(x_opt^4)+(c1/2)*(x_opt^2)+c0*x_opt;

    if abs(c3)<1e-12
        if abs(c1)>1e-12
            x = -c0/c1;
            if x>0
                f=(c3/4)*x^4+(c1/2)*x^2+c0*x;
                if f<f_opt
                    x_opt=x;
                    f_opt=f;
                end
            end

            
        end 
    else

        %Si c2=0
        np=c1/c3;
        nq=c0/c3;

        Delta = 4*np^3+27*nq^2;
        d     = 0.5*(-nq+sqrt(Delta/27));

        % For values where Delta is <= 0
        if Delta <=0 %-> 3 solutions réelles distinctes ou une solution multiple mais toutes réelles
            r3        = 2*(abs(d)^(1/3)); %racine cubique du module de d multipliée par 2
            th3       = (atan2(imag(d),real(d)))/3; %angle(dneg)/3;       %argument de d divisé par 3
            x         = (r3)*[cos(th3) cos(th3+(2*pi/3)) cos(th3+(4*pi/3))];
            x_pos = x(x > 0);
            if ~isempty(x_pos)
                f= (c3/4)*x_pos.^4 + (c1/2)*x_pos.^2 + c0*x_pos;
                [f_s, idx] = min(f);
                x_s = x_pos(idx);
                if f_s<f_opt
                    x_opt=x_s;
                    f_opt=f_s;
                end


                
            end 
            



        else
            d2pos     = 0.5*(-nq-sqrt(Delta/27));
            x      = sign(d)*(abs(d))^(1/3) + sign(d2pos)*(abs(d2pos))^(1/3);
            if x>0
                f=(c3/4)*x^4+(c1/2)*x^2+c0*x;
                if f<f_opt
                    x_opt=x;
                    f_opt=f;
                end
            end
        end
    end
end