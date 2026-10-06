%[text] parameter settings
clear all

P = 3; %4
nGauss = 8; %18
nDelay = 10; %4
GMM_nIter = 10000;

params.sigma = 10;
params.beta = 8/3;
params.rho = 28;
params.eta = sqrt(params.beta*(params.rho-1));
%%
%[text] data generation
[t,y] = lorenzgen(params);
%%
%[text] Visualize
plot(t,[y(:,1)+15 y(:,2)-5 y(:,3)-45]);
axis([0 30 -80 80])
set(gca,'ytick',[-40 0 40],'yticklabel',{'y3','y2','y1'})
xlabel('t')
title('Lorenz Attractor')

figure
h1 = plot3(y(:,1),y(:,2),y(:,3));
%%
%[text] Delayed embedding and data matrix generation
%nDelay = 1;

%ts = [];
%for iCol = 1:size(y,2)
%    c = y(1:nDelay+1,iCol);
%    r = y(nDelay+1:end,iCol);
%    h = hankel(c,r);
%    ts = cat(3,ts,h);
%end

%X = reshape(permute(ts,[2 1 3]),size(ts,2),[]) % Delay 

ts = [];
H = [];
nRange = size(y,1)-1;
for k = 0:nRange-nDelay
    xkT = [];
    for iDelay = 0:nDelay
        xkT = cat(2,xkT,y(k+iDelay+1,:));
    end
    H = cat(2,H,xkT.');
end

X = H.';
%%
%[text] Step 1 Clustering
%nGauss = 4;
options = statset('MaxIter',GMM_nIter);

Sigma = 'full';
SharedCovariance = false;

gmfit = fitgmdist(X,nGauss,'CovarianceType',Sigma, ...
    'SharedCovariance',SharedCovariance,'Options',options); % Fit GMM, GMModel = fitgmdist(X,k) returns a Gaussian mixture distribution model (GMModel) with k components fitted to data (X).
clusterX = cluster(gmfit,X); % Cluster index
if nDelay == 0
    hold on
    plot3(gmfit.mu(:,1),gmfit.mu(:,2),gmfit.mu(:,3),'kx','LineWidth',2,'MarkerSize',10)
end
hold off

% figure
% gplotmatrix(X,[],clusterX)
%%
%[text] Step 2 Calculate mean vector
acT = gmfit.mu; % Q x 3(d+1)
ac = acT.'; % 3(d+1) x Q
%%
%[text] Step 3 PCA (LSUN) design
for c = 1:nGauss
    c
    Sigmac = gmfit.Sigma(:,:,c);
    [PsicT,latent] = pcacov(Sigmac); % <- This was wrong.
    Psi(:,:,c) = PsicT.';
    %Ic = find(clusterX == c);
    %Ic = Ic(1:end-1);
    %
    %X0cT = X(Ic,:);
    %Z0cT = X0cT-acT(c,:); % Mean removed
    %Psi(:,:,c) = pca(Z0cT);
end
%%
%[text] Step 4 
%P = 2; % # of channels
for c = 1:nGauss
    Ic = find(clusterX == c);
    Ic = Ic(1:end-1);
    %
    X0cT = X(Ic,:);
    X1cT = X(Ic+1,:); % Advanced (1 time step)
    Z0cT = X0cT-acT(c,:); % Mean removed
    %
    Psic = Psi(:,:,c); % PCA coeffs (loadings) for the specific cluster
    PsiZ0c = Psi(:,:,c)*Z0cT.'; % (PCAcoeffs) x (Mean removed data) 
    SPsiZ0c = PsiZ0c(1:P,:); % Truncation of channels
    Q_bc = (X1cT).'*pinv([SPsiZ0c; ones(1,size(SPsiZ0c,2))]); % Advanced data x inv(truncated)
    %
    Qc = Q_bc(:,1:end-1); 
    bc(:,c) = Q_bc(:,end);
    %
    Pc_ = Psic(1:P,:)*Qc;
    %
    %[U,S,V] = svd(Pc_);
    Pc(:,:,c) = Pc_; %U*V.';
end
%%
%[text] Learnable parameters
%[text] ![figyy.png](text:image:22f8)
%[text] $\\mathbf{S}\\hat{\\mathbf{\\Psi}}\_c$
SPsic = Psi(1:P,:,:)
%[text] $\\hat{\\mathbf{P}}\_c$
Pc
%[text] $\\hat{\\mathbf{a}}\_c$
ac
%[text] $\\hat{\\mathbf{b}}\_c$
bc
%[text] Classifier
classify = @(x) cluster(gmfit,x.')
%%
%[text] simulation
%x0 = rand(6,1);
x0 = X(1,:).';
K = 1728;

% Hard decision
x = x0;
Xhat = x0;
for k = 1:K
    c = classify(x);
    ac_ = ac(:,c);
    bc_ = bc(:,c);
    Pc_ = Pc(:,:,c);
    SPsic_ = SPsic(:,:,c);
    x = SPsic_.'*Pc_*SPsic_*(x-ac_) + bc_;
    Xhat = [Xhat x];
end
Xhat
%

yhat = Xhat(1:nDelay+1:end,:).' % <- This was wrong.
that = t(1:K+1)
plot(that,[yhat(:,1)+15 yhat(:,2)-5 yhat(:,3)-45]);
axis([0 30 -80 80])
set(gca,'ytick',[-40 0 40],'yticklabel',{'y3','y2','y1'})
xlabel('t')
title('Lorenz Attractor')
figure
h2 = plot3(yhat(:,1),yhat(:,2),yhat(:,3));
%%
x0 = X(1,:).';
weightin = @(x) posterior(gmfit,x.')
% Soft decition
%x0 = rand(6,1);
x = x0;
Xhat = x0;
for k = 1:K
    w = weightin(x); 
    %w = posterior(gmfit,x.;);
    x_ = 0;
    for c = 1:nGauss % TODO: to process in parallel
        ac_ = ac(:,c);
        bc_ = bc(:,c);
        Pc_ = Pc(:,:,c);
        SPsic_ = SPsic(:,:,c);
        x_ = x_ + w(c) * (SPsic_.'*Pc_*SPsic_*(x-ac_) + bc_);
        %x_s(:,c) = x_;
    end
    %x_s = x;
    x = x_;
    Xhat = [Xhat x];
end
Xhat

%Visualize
yhat = Xhat(1:nDelay+1:end,:).'
that = t(1:K+1)
plot(that,[yhat(:,1)+15 yhat(:,2)-5 yhat(:,3)-45]);
axis([0 30 -80 80])
set(gca,'ytick',[-40 0 40],'yticklabel',{'y3','y2','y1'})
xlabel('t')
title('Lorenz Attractor')
figure
h3 = plot3(yhat(:,1),yhat(:,2),yhat(:,3));
%%
function [t,y] = lorenzgen(params)

sigma = params.sigma;
beta = params.beta;
rho = params.rho;
eta = params.eta;

A = [ -beta 0 eta;
    0 -sigma sigma;
    -eta rho -1 ];
v0 = [rho-1 eta eta]';
y0 = v0 + [3 2 -4]';
tspan = [0 30];

[t,y] = ode45(@(t,y) lorenzeqn(t,y,A), tspan, y0);
end

function ydot = lorenzeqn(t,y,A)

A(1,3) = y(2);
A(3,1) = -y(2);
ydot = A*y;

end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":32}
%---
%[text:image:22f8]
%   data: {"align":"baseline","height":322,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1sAAAFCCAAAAAAiEZijAAAPIUlEQVR42u3dCZbrqBJF0Tv\/SXlo+r\/yuVEvQAGEghNrVTYvXRYCtmmEhCbCSYhz56SIKiWhkc9d2CLqyRoWV9Rzx5aXNkuDlkXcc8eWo5IQ544tok5JiHPHFrHs1BgVhMal9T73SAMvbBnYsqgQzMCbZSW2Qtm6XSWGLgcZZiS24tkS5WBCC1vExta9WjG8LZNcxFZUWzfqxehdQos8xFZMW6+bFUPQ+uQitoilrdc9XRo8\/2Z5iC1ibeuOLmj9chBbxNZWOS5BC1vEma1iXUIWtohzW2W6hCxsEZe2SnAJWtgirm0V6BKysEWk2MrVJWRhi0i0lYdL0MIWkWorS5eQhS0i3Va6LiELW0SWrVRcGqsCnNDCFpFoK1GXOqfdjSxsEem2UnSpZ8IbP6DiXBa2iBxb17h6dwnlhxa2iBxbl7q6D7fkRRa2iExb57oczGTIiSxsEdm2znANYiuJFraIbFvHujRE8afJwhZRYutI1wgXt1JlYYsos7WPawBb6bSwRZTZ2tMVn1aGLGwRxba2uqLbypKFLeKGrTWu4LYyaWGLuGFrqSs2rVxZ2CLu2ZrrimwrXxa2iLu2frgELWwRlrZeLw\/7CsibLGwRBrY86JI3WdgiTGx1x+WQFrYIE1u9dcmdLGwRVra66pI\/Wdgi7Gx1xCWHtLBF2Nnqp0v+ZGGLMLXVSZccysIWYWyrCy55pIUtwthWB13yKAtbhL2t5rrkURa2iBq2GuOSS1rYImrYaqpLLmVhi6hkq6EuuZSFLaKarSxcqh3taWGLqGbrretptkzPHltEHVt\/9etZtmzPHVuEB1uveoEtbGELW7UK9f1Mk+JEYQtb2Do6tff32+2WhVRsYSuSLd2htWy3bkvFFrbijLfWe9bmJk6mUrGFrUBzGZun9+v81afjrbtSsYWtSPOEq+Sc4EnY\/ylPKoGt6LaUZuu63bqUSmBrMFtK7\/Rd2xK2sIWtf1o2tuYJLLAluoTYwtbH0mJDJ02Hu6ddj7fOpRLYGsfWqiP3z1WprUupBLZGsfW92Pv74c5461Iqga2cVHwX1T9zPeF0sFxp946Bc1vXUol7ttTDllra0h1crhfbbcdHye3WuVTCwpYS6rm5Lb3UztbOwaLsd\/yxVTqXcSWVuGNLKY2ItS0VtVyFqdg7WBBbn3m98jn4C6kEtka1tR0mrdc8ZSy82JFK3LClpNGPsS2VDbnKUqH9IVeMPuERt6I\/M6HBPKFBKrBFYAtb2ComUKXssOXTluLuKO2Uln1m79r6XNPR9krPRV0zq9XL65sNbB0cbT1Lv8mZKrZUr8CxdZjTUnVb73\/Y\/CWljlu2W78E5PIqvr6l8+tbq5xZYbQrnJoF7stWt\/NbLFKZNLtCbvf222r7\/b2OLSnfVi6u0lTs2TrLmSWu+2UzW1pUocDz3kgNG40+7dWP1jT\/uYmtVyVbyreVias0FZfrC0\/zyMrWd2GR8Qd73jupHa1+uBZZbZ7Xpz0\/Laq1jjqLeb2xz2F72\/qlIvEMNzlTwZZmRVynwF3aUhdbWmX1ZNopVIKt7z8tv92xpe62VqlIOsP6tr6fO1MVW1l1Ry1pdWq4tMxq67zes7XtJVraSt6Zp6qtaT0NkWTrdK7DomT2C7zHzIHSJloeH2tbxrE\/+11qq+yo1rbyUpF2hhdz9A8pcGy1jKOK6cvWy42t5edODVv9t0+i3aph61VY8yraKri+VcPW3vXl8dqtIHMZ0\/5chvXV\/6ub2e3HW8lzGWULM7JTkXyGV7YsL\/jUmbxinnA1S15v2kjnHaB51Zt\/bT0Hb25rlYr0M1zmTKV5wjHm4KfetH6bSLS5vnXQJzOzVXbt2NrWOhU5tpYrRizXEy4fRlFnsYAnW5OndRkN1jxp74LO36ve\/+lurZ5c2FqlIuEMdbGAw3bNk32By5+tfjdYnj70qqmtgwnoQLYSzrCtrUrlja2zaU\/7tz6+GHqx2m53Aj\/1sIW2UsQZroNPypmq6+B71blBbFVc1Xi25umq5v33+\/Y1dW3tHLOdLTWy1XkZ61i2pmqPWzxe83S09Gn54d3Y1t4xa9parw25WhNl\/XHasb5hy97W0X2Rm+GI2tx3fHg7VYP7jhetpC66pBVsTdh68GzlnQmD9s\/LUD1bh4dLf+5whedlYKvhgAtbLW3lzk7KfXljayenp0bzhLn1XB1sNXoCqXrZ+hY184RtErC4b86Brff4o2m7tXvMmLaW65+w9cBU3Hk+4f4a+qq2Xqkr6A1tdRlvda912Opqq\/14q3EqlDXowha2sJV1JeCFLWxhq28qsOX\/LA\/uFMUWtrBFu4UtbGELW9jCFrawhS1sYQtb2MIWtjrZ+i40ko+FT9jCViBb6r8\/JrawFbFPKB+ysIWteOMtJ7Swha14cxk+aGELWx5tPWTriLFsOchxbGHrMbb0OFvCFrYeYCuzVvuwZXv6Q9p69YtBbNnO+TQcSltmALawZW\/L+uHl0+NwYQtbNWxZX6hoV63sdGELW\/a27C8BtqxWVriwhS1zW8+mZdZ0YQtbxrZqrFtpXatMdGELW6a26qwIa1+rDHBhC1uWtqLQsmi6sIUtO1vVNlfpU5tu6sIWtqxsVbtBoN9eibdwYQtbRrbq3XvTcWOcO7qwhS0TWzXvauu6XWK5Lmxhy8BW1ftFHWx2VlRfsIWt+7bq3ord+z7U0qYLW9i6a6vyQw687NtelC\/Ywla5reqPD3HyLMh8XNjC1i1b9Z\/M46HdKtKFrW1N+fvhlbmR2fJNRrHV4JlXLpqtorYLW1sU\/\/\/62dlW+aD++361bW4QW02eJicntPJnDLE1z4s1lCJbr1FsjULrl4a8pgtbO7ReBe1Wxv8Sw1ajR6C6spWnC1t7tEps7b1NXFvNHi4sV7SyOobY2p+AwBa0DtKQ3HRha98EtjzIcmkrWRe29ufNsXVFaxqWVmrHEFtn16R+8+rzv\/\/9ouWP7xd8nuI6u1C2e6wH22q6jYtXW0lNF7YSbC0bMf1+0Xbe4\/s+i5eHsdV4gyS3tlJ0YSvF1vzb\/DLWTv8xtq3GW485ppXQMcRWLVvLl4ew1XxXP9+2rpoubKWMtxZYNGuaFqOw4Lba75fpgdZ5Ik51Yetgfk+rf\/zloXZ\/DG+rw0608k7rvGOIrbNJ+Nk84Xq94WYKcbuU42\/VbpA5+C6bPD\/B1knTha2D\/t\/leEuLH2Pb6rN9uv8u4bkubB0sKNyqeVvR7xXaaeOWK34V4tpxF1kPabZOOobY2sW11yJ9bP1mB3eueAW0pYFppSZit+nC1s6Ya32N6vP1bWt+Y\/LJHPzh+qdH2eol61m2dnVh6+Cm\/sUs4PsHvUdQsxv\/F6\/Qa5XDz7fVTdaTuoQHHUNsVX3qzMNtaXBaeYlYN13Yak6rlS3dtdVT1hNtrXU1tlW5pPzYkvraKi7X1ZU79arjT6S16hg2tVX9U9CXLfW6f+vvLzdtfRstbJU2Xe1s\/fsMHMXW63wNZ4PPscLq+Z21edeMkbuE5Z9O30xs1yccyFbv+45v2ZoNtGi2SnU17hPWzjRsLXJCxfXiVy2wZZCJ2MLWrFpM0NK9wmlqq34nA1u3q8WqSmDLKCOb2KLd8m9rwtb9RLS05ehJkdFtyepdoVVxZg9bo9nyUMmD2LI9kTrlja28vNbD67gLWk5SgS1s0Wz1s+Xq6eGxbWnCFraw5doWXcJH2PL1aP7Qtmi2QtLCFrZothrbcrbtRWRbNFvYwha26BJ6Km9sXeS1nl6\/aLawFd4WXcIn2HK3FVpcWzRbUWk9yFbfwBa2vJU3ts7z2pIWtrCFrUDNFrZ6lbeXLJHTvNaELWxhC1vY8lTe2DrNa03Ywha2sIUtV+WNrbO81oQtbGELW9jyVd6xbfma+MdWCy3ur\/REseUq0dh6+lGwhS1sYQtb2MIWtrCFLWxhC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFLWxhC1vYwha2sIUtbGELW9jCFrawFcmWsIUtbNU4isrS8xRby9s4sYWtdkdRYYKeYeufK2ELW9iybrN+wFzYErbGsaXSFD3D1uTLllp8jGHr6Ud5gK1mj+FWxuuELWxhC1vYwlYqrb62Svvf2HqerVuPGcSW148xbLk4yiC2mIPHFrbqvKkTW6X9BGxhy52tzRR8T1v\/8rokx7GFLX9ZsmkputsqyXJsYesBBdPfVkGeY+uxtspGAdjKfVdN2BrLVukoAFuF74qtwfqE1QscW9\/PMmwNNt7CVv0+YWkfAVtPtyVstZ3LSG7BsIUtbOV0v\/9NI2ELW9gytvXuH2ILW9i63yd8v3b9nAFsYQtbN9\/1PUf491XYGsjWlDfCxtatd8XWMLZmPySWO7YsbDHeCmxrWl5zSR9hY+veu2YMurD1UFvLoZY6ncSAz9VNX6CBrSfaujEKwFaz1GMLW9jCFrYMRtjYwha26oywsYUtbNUZYWMLW9h6wttjC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFLWxhC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFrWa2Kge2vNlyVeChbfk6KLZIBLawFTIVvmiFtqUJW9jCFrawFYxWZFuasIUtbGELW9FoBbalCVvYwhZHxFY4WnFtacIWtrCFLWzFoxXWliZsYQtb2MJWQFpRbWnCFrawhS1sRaQV1JYmbGELW9jCVkhaMW2NVcmxhS0qObZGohXSliZsYQtb2MJWUFoRbWnCFrawhS1sRaUV0NZwhYwtbFHI2BqJVjxb4\/X7sYUtbGFrJFrhbA14mQVb2MIWtkaiFa3Oi+rFyWMLW9gKTSuYrSFvIsIWtrCFLWxhC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFLWxhC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFLWxhC1vYwha2sIUtbGELW9jCFrawhS1sYQtb2MIWtrCFLWxhC1vYwha2sIUtbGELW9jC1vwYf9EtJ7scve85d08FtpocR5\/oWLuaH73rOTtIBbba2lJXW4p\/VEepwFaL46hrPVOXw0secPVMBbYa2xK2sDWSLYKoENgK\/xlOu+W\/qmMLW9jClltbwxwdW9hqXMTjHH34OXhstS3ikY4++rVjbLUs4sGOPviaJ9fxPxvbkeYHCOtoAAAAAElFTkSuQmCC","width":859}
%---
