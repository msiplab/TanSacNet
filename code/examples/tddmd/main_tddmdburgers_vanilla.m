%[text] # **Analysis of the Burgers equation by vanilla DMD**
%[text] First run TanSacNet/code/setpath
%[text] Requirements: MATLAB R2022b 
%[text] Contact address: Shogo MURAMATSU, Faculty of Engineering, Niigata University, 8050 2-no-cho Ikarashi, Nishi-ku, Niigata, 950-2181, JAPAN http://msiplab.eng.niigata-u.ac.jp  
%[text] Copyright (c) 2023, Hayato Obara and Shogo MURAMATSU, All rights reserved.
clc, clear
close all

% Data folder
datafolder = "../../../data/";

%[text] Parameters
nu = 0.05;
% td-DMD
nDelay = 1;
%%
%[text] ## Input data
filenameinput = "burgerseq";
if exist(datafolder+filenameinput+"nu_"+replace(num2str(nu),'.','_')+".mat","file")~=2
    main_burgerseq_gen(datafolder+filenameinput,nu)
else
    disp("Data found!")
end
Sload = load(datafolder+filenameinput+"nu_"+replace(num2str(nu),'.','_'),"T","X","DataT","dt","dx");
T = Sload.T;
X = Sload.X;
dt = Sload.dt;
dx = Sload.dx;
DataT = Sload.DataT;
%%
%[text] ## **Data visualisation**
figure
disp_imagesc_(T,X,DataT," $z(x,t)$ of Burgers' equation");

%%
nT = size(DataT,1);
nX = size(DataT,2);

%[text] ### vanilla td-DMD
% Construction of Hankel matrix
x = DataT;

ts = [];
H = [];
%nRange = size(x,1)-1;
nRange = size(x,1);
for k = 0:nRange-nDelay-1
    xkT = [];
    for iDelay = 0:nDelay
        xkT = cat(2,xkT,x(k+iDelay+1,:));
    end
    H = cat(2,H,xkT.');
end
H
%%
%[text] Step 2 SVD of H
[U,Sgm,V] = svd(H,'econ')
%[text] Step 3 PCT
iMode = 1
ui = reshape(U(:,iMode),size(x,2),[]).'
V0 = V(1:end-1,:);
V1 = V(2:end,:);
dtAplusI = (V1.')*pinv(V0.')
%[text] Step 4 Discover forcinge signal
v0hat = V(1,:).';
vk = v0hat;
dtrk_ = 0;
dtr = [];
for k = 1:nRange-nDelay-1
    % True value
    vkp1 = V(k+1,:).';
    % Forcasted value
    vkp1hat = dtAplusI*vk;
    % Forcing signal
    dtrk_ = vkp1 - vkp1hat;
    % Update
    vk = vkp1;
    dtr = cat(2,dtr,dtrk_);
end
%%
%[text] Learnable parameters
%[text] ![fig02a.png](text:image:51f8)
%[text] ![fig02b.png](text:image:3387)
%[text] 
%%
%[text] Parameters
%[text] $\\hat{\\mathbf{v}}\_0$
v0hat
%[text] $\\{\\Delta t \\mathbf{r}\_k\\}$
dtr
%[text] $\\mathbf{I}+\\Delta t \\hat{\\mathbf{A}}$
dtAplusI
%[text] $\\hat{\\mathbf{U}}$
ndim=nX;
Uhat = U(1:ndim,:);
Uhat
%[text] $\\mathbf{\\Sigma}$
Sgm
%[text] 
%%
%[text] Step 5 w/ Discovered forcing signal
vkhat = v0hat;
%dtrk = dtr(:,1);
Xhat = (Uhat*Sgm*vkhat).';
for k = 1:nRange-nDelay-1
    dtrk = dtr(:,k);
    vkp1hat = dtAplusI*vkhat + dtrk;
    Xhat =  cat(1,Xhat,(Uhat*Sgm*vkp1hat).');
    % Update
    vkhat = vkp1hat;
end
%%
DataHatTaprx = Xhat;
%%
figure
ax = disp_imagesc_(T,X,DataHatTaprx," $\hat{z}(x,t)$ via vanilla td-DMD");
exportgraphics(ax,'fig03b.png')
% MSE評価
mse(DataT(1:size(DataHatTaprx,1),:),DataHatTaprx)
%%
%[text] ## Discovered forcing signal
r0 = zeros(size(v0hat),'like',v0hat);
%dtrk = dtr(:,1);
Rhat = (Uhat*Sgm*r0).';
for k = 1:nRange-nDelay-1
    dtrk = dtr(:,k);
    rkp1hat = dtrk/dt;
    Rhat =  cat(1,Rhat,(Uhat*Sgm*rkp1hat).');
    % Update
    rkhat = rkp1hat;
end
%%
DataTdfs = Rhat;

figure
orangered = [255 69 0]/255;
ax = disp_plot3_q_(T(1:size(DataTdfs,1)),X,DataTdfs,orangered);
ax.ZLim = [-3 3]*1e-14;
exportgraphics(ax,'fig04a.png')
disp("||q_k||_2^2/(K-D)")
size(DataTdfs,1)
norm(DataTdfs,'Fro')/size(DataTdfs,1)
%%
%[text] Function definition
function ax = disp_plot3_q_(T,X,DataT,c)
if nargin < 4
 c = 'blue';
end

for idx = 1:length(T)
 plot3(X,T(idx)*ones(size(X)),DataT(idx,:),...
 'Color',c,...
 'LineWidth',1)
 hold on
end
xlabel('$n$', 'Interpreter','latex')
ylabel('$k$', 'Interpreter','latex')
zlabel('$[\mathbf{q}_k]_n$', 'Interpreter','latex')
ax = gca;
ax.View = [-30 30];
ax.PlotBoxAspectRatio = [ 2 2 1];
ax.FontSize = 20;
ax.TickLabelInterpreter = 'latex';

grid on
hold off
end

%[text] 
function ax = disp_imagesc_(T,X,DataT,title_)
imagesc(DataT)
colormap("jet")
cb = colorbar;
cb.TickLabelInterpreter = 'latex';
title(title_,'Interpreter','latex')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
ax = gca;
ax.PlotBoxAspectRatio = [1 1 1];
ax.XTick = 1:20:100;
ax.XTickLabel = round(X(ax.XTick));
ax.YTick = 1:20:100;
ax.YTickLabel = round(T(ax.YTick));
ax.YDir = 'normal';
ax.FontSize = 20;
ax.TickLabelInterpreter = 'latex';
end
%[text] 

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":40}
%---
%[text:image:51f8]
%   data: {"align":"baseline","height":179,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtIAAACzCAIAAAArP2wrAAAOiUlEQVR42u3dCY7jNhBAUd\/\/UjmaAyTApNO2qOIqLu8jCAZu2ZaqSNYXLVGvNwAAwBBeQgAAAGgHAACgHQAAALQDAADQDgAAQDsAAABoBwAAoB0AAAC0AwCARNl7KXy0AwCAUc7BPGgHAAAjnOPz36AdAADQDtoBAMDizsE8aAcAAKAdAAAAtAMAgJsi90Hir46CdqBNYxUHAEbCq8FwlUFyk6PQHE\/ubEDlKRc244Smu+652SZHYSQ9qk4ICGgHDtSOxDlY86PuGvlhR0E70LIDCAtoB2hHYpJg2phkHcWcB6ICHVonRAYNzxchs4seZsI5pp1ICB7FtAdi+Dj69FR8oDiBdnxudvvKKtox4YEYPk4ZTf76B\/IBxQknZ\/bnka5VrXOPgnZgCu1gHlCcQDsSBbv+PpEBV9UE5WnOG14MH8dpB\/mA4gTakb6YdMJqnXUUtAPTaQf5gOKEMzO7aLXOTRntwIzawTygOIF20A7agXHaQT6gOOG0zK5YrbOOgnZgdu0gH1CcZFZm19KORQ9EI6MdzAOKk8zKbGoWYdHIuIEWC2gH+YDiJLOi8WsxrnW1Y8ID0choR4Z8iKfmpCXI7FEBWT0sEx6IRkY7THtAcZJZmQXtwATaQT6gOMksQDswVDv85gLFSWYB2oGh2mHaA5IuswDtwDjtIB+ak3TLLILxFAfaQTuYBxQnyKx40g6sph3kw2AKmYUJD9qBodpBPhQnyCyElHZgqHYwDyMpZBaJqAos7aAd5AOKE74XSHRCM6MdtIN8gHaAdtAO2oGVtYN50A7QDhgDaQftGI2ORzsgs6IqsLSDdpAPKE6QWSGlHdhRO5iHkRQye2xIxYF20A7yAcUJMiuetAO7awf5MJhCZk11QFxoB\/OA4iSzMgvagU21g3woTpBZ0A7QDvIBxUlmAdqBfbWDeShOkFnQDtAO8gHFSWYB2oFNtYN8KE6QWdAO0A7mAcVJZgHagU21g3woTpBZ0A7QDvIBxUlmAdqBfbWDeShOkFnQDtAO8qE4QWaBgdoxrI3qCbRjNvkY\/NVTdQHFiXY40nmO4iDtGH+0RjraMYl5PNUOJ+kCtEMxXv0wC97znjUsi6bsNSJti4\/4tIN8TNICZ9gB2kE7DnKOubVj0cq4mHYY72jHg\/IxSdVXnCCzQ6c6CsxjYCSXS9xrocMz3tGOB81jhuZHOyCzQ\/tOmXaMnSChHYJLO56XD9qhOEFma\/vOn+1pB+0A7UjLB+1QnCCzjbUj\/nbaQTuMJidox0\/zoB2KE2S2mXbkmgTtGKYdrztoB+3YTzvSTX2zLkA7aMcp2vFr46wJj4\/Nvo4Dx1bGLtrRqTUb72jHzLMdV9\/ecJdoB2T2Ge24ncP4efHpxYWoP4P85\/\/1YT9dOxKKRztoxyHa8Tnb8djQqThBZgv6ztctgxMe19t0mvDYXzvif007R1m4y043jSa0o0kkc8fugtnBybtA1ue\/Mtmgl5UdxQxBOEE7qqY6IhMeYe0Inq7vWhlHaEdDR7vdnnbQjge14515McdyXaBAO8peXLEKVu4w7ZhFOyK\/pOQ7R1aQ22rHbJXx1SVtMX8vPjNoprS0g3b0bPyNz8Am6AK52hEf\/mgH7Zh9qiMy4dFOOzaujK8ezTF4tlcciPRvN1SDdjw7FmfNdqzVBepP1+Kv0w7a8cBBBX9DKfoVZsAJyRKVsf1sx9WY22pMafurDe2gHV2n+no01we7QP24mXj74c8ipx1TzHZEtuysHWWJWKgyvponL3EbbeKVJg7Y9SYC2uHajqzikasdS3QB2kE7djaP+LIcX\/Xi88Winw4LevFalbHL5WxB7fj17ybBLfhM2kE72jp3cJRZsQvQDtpBOy5nNdppR24vXqsy9lql9PaNnwcfXBfhNrhZn0k7aEfDqpB7D\/lyXYB20I5ts1PwsLevK5lef2CWduTeir9QZey+SunX12uu1Du229COJZYLy5r\/W64L1H9FfJI5scx8k9hWrhry+fa0Ud1+1237iazNH7de2nFpEln\/XblL6WpVxfdwLJTZZqNe1lpAZR0+64kwhIN2jBm24g9kuSqZC3WBMdpxW4Y\/541\/BrbsBLEyEbdmED+0xPRPwf1BZjtC+S1zjqsJjzt9LPPIbSrjM0+gLbu8Jb4KCPOgHQ+cLR3QBSbRjuDrWUW6Pgs1y6PV2AntGN9\/u+5n7j4vVxnn0o6aH7bLPpN20I6ptGPyLjDPbEePYtzjspX4xjXaEZwsoR3rakfxtR0TVsYHtOPqZp6au5mLP5N20I7xw9a6XWAD7ci9uqLm7a02ph2HaEdZL16uMj6vHe8Wi6js8UyHQ7Sj384sqh0LdYFttKP4MXVlKYhf8XYbedpxjna8c65CXagyNl74fdjsExbVjn\/3pNP+DHj+6jwTsM82py1nOxKu0HUCIyJJuTtQvKL\/3oOh45pih1c5SM6xgXb83I0euzTmDtKTB5cTtKP5hFPlmW7ZDtCOQ+rIikf0emr00Rlox4raMb5NznXn29aXlL7Da9Fm7XDwc4KzGmm3qF+G5Jwh0VEsph2gHfWXdDTfK5JKO4o3fne4qPNd8cRd2oFthxEhoB3uZMFR2vFu8YNIPDJtlwt7\/39JtOLw0g7QDtAO2rGndsTv5rja\/n29GmzWxokvKg7F1y\/N+q74QreRLNAO0A48UBhoB1o1J8Vp6ZTRjhnGYdAO2lFb13995u3n3y5gQDuWmNsQ4S2dQ2ZpB+1As1LRqq5\/\/ZAyb+g3AWPoHNacRGba0wzaMaeyg3bs39x\/\/d7c3DmKHYJ2bNCcRGZC7ai\/1FdmmQftQG2p6K0dt3+lHVs2J2HZrLzJLPM4SDteGEJ9Lc9dk+PqjbevF++tLA\/D0HmCXEKXoR1YRjsS742\/SDuMoTAO6zW0Q3M\/VDsKPod2GEBBO6DXzKUdGDOI9Cjn02qHBtC7OYmJqxCQ1XFEg3acMoIMmEWgHQc2J2GhHeActANfho+263bkygft2LU5iQztgJDSDvzX1j+bfo9VSmnHsc1JZNRIiCftQHftSCtI0Dwir9COOYdOEVYmUTAOg3Zs3ty\/jia9n75GO45qTrMVy7I7CMoulf36ltXvZdB3aAftQOPRpKaWVy7s8UtNvv67iRgZOo8tTom9iuxw\/LiCKrNcKdJ3QDuwj3bU34hLOxSnypPL+LRHP+2YWT70HdAOTKQdkbtkPZOFdsysHbfzEEFjCBpM\/d7KLGgHjtaOl0fBgXbQDoB2YJh2tLrrlXbQjge14+tmv64MHaMdU4VR34kHauOvox2YRTvqnxPb75ExtENxis8f1GhHXE1q9kRm53eOYVEa\/HW0A7Nox0IYOmkH7aAdAyYeBgRq8NfRDtAO2kE7emnH55afS6KN0Y4JV0DRyGkH7QDtoB2042HtiHgJ7TjEOQaowOCvox2gHbSDdtAOmcWmw4gQ0A7aAdqR+LT0X2kHGTpzHKAdoB20g3YUbvaUdshsepcK3vPW\/aeXDxmiHbQDtIN2TDf7Uva23trRNkpZDw5s8nDBVxFtM2WAph20A7TjdbVl4k+0Y8apjgLzyNm+U5SyTKJyBwqez9w2IAZo2kE7cIR2RNbkGKMdrqLoEp8y7cjZvmaaYTntCDYM2gHaQTtoRzPtSMx10465tOPP9rSjj3nQDtAO2kE7Msa+yMNWgm\/x4PsFtCP+9vDG9ddVzKMdWeZBO0A7aIfilDeTEfwNu7d2zNw+99GO3AmPHO3oN+ExXjtyzYN2gHbQDtrxrr8IP7H91xtYrrav3BOZrapkvzbOmvDI+YWFdtAO0A7aYbYDtOOVN4fx8+LT2IWoV4q5tHa8My\/yoB2gHbSDduB47fi6ZXDCI\/MXlv20o948aAdoh6JIO7C8dlRNdcSVIqYdn5Fpbh4PakcT86AdoB2gHThAOyK\/pDSd6theO7prIu2gHbQDtAPzZLbNVEdELEqnOnqYx7Pa8a5\/ulvWSq\/6G+3YSTgURdqBFTNbsgPB31BqfoUZdY\/SDLMdld+Ysf\/6G+3YSTgURdqBU2Y7Ilu20I4BQXv8ktLKbzTbgVO0g3PQDuyX2VfQJ4Jq8lUvPl8sXRquSdyevYG28htd24EjtINwKE6gHRnmUaodrVyhVQqaX8Rarya0A5trB+FQnLC9drxurxXN+bjvkvFrPfWimYDbUSgrqvWPe80yiSZbWrcDm2sH51CcQDsulxlN\/3flLhXGkB6ICoapq7c0caDcaZX6B8XRDtpBOEA7sLh2lDnH1YRHck2O9GMCIze2VK7+WWwtwY8NQjtwlnYQDsUJtGP18O76dbQDu2kH51CcQDtoB+0A7SAcihNklnYc7Ry0A5toB+FQnEA7aAftAO3gHJqTjMjshrWTc4z8RsMH7SAcoB0yO11mNbNdpUpeaQfhAO2Q2Rkzq71t2Vqkk3ZwDigDMiuzGNXYhIB2EA4oTjIrs6Ad2FY7CIfiBJkF7QDt4BxQnGQWbeIpDrSDdhAOKE6QWdpBO3CAdhAOxQkyK54QFNphkgMGU5mVWSGlHVhfOwiHkRQye3hIRZV20A6\/qkBxgswyD9qBXbTDJIfihHULJDqhmdEO2kE4QDtAO2gH7cCa2qGn0Q7QDhgJaYfRpLt26Ga0AzILUaUd6K4dhMMwCpmFkNIOjNAOzmEkhcxCPGkHumsH4TCYQmaRjqc40A7aQTigOEFmaQftwCLawTk0J0mXWYB2oLt2EA4oTjIL0A501w7CAcVJZgHagRHawTmgOMksQDvQXTsIBxQnmQVoB7prB+GA4iSzAO3ACO3gHFCcZBagHeiuHYQDipPMCgVoB7prB+GA4iSzMgvagRHawTmgOEFmQTvQXTsIBxQnyCxoB7prB+GA4gSZBe3ACO3gHFCcILOgHeiuHYQDihNkFrQDI0YTwgHFCTIL2oFntENk0NZisQcaOWgHGtcJMQHtAO0A7UD3OiEaoB2gHXiWvwFb6sHStEeejQAAAABJRU5ErkJggg==","width":722}
%---
%[text:image:3387]
%   data: {"align":"baseline","height":247,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAw8AAAD3CAAAAAB+njfjAAAMVElEQVR42u3dAZakKAyA4dz\/UnU0dt90t4UYBARC1D\/v7W5vTZWVsfnEIKIE4q0h\/wfZJd9Ks3ixBr8eVmWHh1dzENfJ4YGwbXBCdnhAw29r8+lBBA\/EilMljx5+k\/t88EAYFw4OPWwc8EBYl9HuPHw14IGw5uDNQ6wBD4SxBm8e9hzwQNhq8OUh0YAHwpqDIw8HDXggjDU48qBwwANhq8GNB00DHghrDj486BrwQBhr8OEhxwEPhK0GDx6yGvBAWHNY7uFEAx4IYw3LPZxywANhq2Gxh3MNeCCsOaz0UNKAB8JYw0oPZQ54IGw1rPNQoQEPhDWHRR6qNOCBMNawyEMlBzwQthqWeKjVgAfCmoO9h3oNeCCMNdh7aOGAB8JWg7WHJg14IKw5mHpo1IAHwliDqYdmDnggbDUYemjXgAfCmoOVhysa8EAYa7DycI0DHghbDTYeLmrAA2HNwcDDZQ14IIw1GHjo4IAHwlbDdA89GvBAWHOY66FPAx4IYw1zPfRywANhq2Gmh24NeCCsOUzzMEADHghjDdM8DOGAh9c0ZicaVA\/iREOXB8HDnTiIEw6Kh1HJfT4LPfT8JfCwoHcQFxoUD73JDeRw3UPXX0JQcisPAzWM9zBQgysPEtav6zkx1hcP0rdvwiQPfcmFkRque+jcwxO6dDxM3TdDNzc8u89qD+MHO\/Dg2IPvzX0+z\/PQ2WuO8HCfJnXt3N\/JDQvJs9l\/k+qZPv5EDw7OCZ7pIf5y8ZB85nHUHdPH13voPA\/Aw5K\/mzj1IB0c3Hjo2VMyvE\/HQ8W3y7VsJtcP\/\/7\/+1rrt\/nw0NeIZHifjoeao9f3hfO329bTyTGw+DgVrx6+n202PbxPx0P56yXfP2cbp8n40v6Vs9lNenZ+PFw74KjjS\/s+fcWit3hY0z8kO78w28+lh\/Rv0XbAOfHwPW3Cw+Dvl9rj0xIPJ4fWe3jYfbJtB+vX4\/bHsAXPCHiXh\/wh2b5+OJxqnJxAOfWwDQhdOODgYUV5tD96Scgezow9\/F2OyyV3Gw\/f7BoPOJn5fPExTPAwL4Gf5ubGQzI2kyZ3Cw+y99B2wKn1YPyggLd4EFf1g0jSoO5YPyTDo40HnKyHfYEuwfZJAU\/2kBnhUOccGnrYT1\/6aUTRqVNldm6uTwdJTv56POx31KHXxMNIGskxeu340nlyN+gfErStB5yyB6F+mO9BPHu4Wz3dc8DJDR5JZjwED2Nz+R3j9Ng\/KMnd14N0ejjtNZ17kHt5CLlLwsq05fMGPMFDfgKHnl2pgRs\/b7f5gFPv4Tb1Q+Hmb+f3x13HfOGm9\/YHzpU2lzTpHhDDPLQccIr7UOs1XXso3ez6VA9XbvKd7EE+cmzkq++Pk64\/Dn13V+DhxR7k2EPc38OKk2rpXm9QbuvhagO+dNP73PUE8PC08aWrd+JOW8jjTutriF5CsJ7AbT1cXKNm4tI2t19vBg\/NyZT2w\/RngOhzFCrHMpo\/1rpvXHrYqkyXHsSlh4rmsc2f6fIgneOt0bS2n3HmupYdf0xC10JG2bQzN\/BINIltP1Xh3OZbPIg\/D1UHzG0aWdvplIz2EDfmqj4r2nD6sXEgcslL5ppZcq\/bXA+Seogc5j38\/bFM9iB+PRTberOH46Ip0utBkrus6kG0f6zll3rdgz59e3h2hyZ91j\/8\/s8fmcTPoz1UnlGX3yPatJmxHjaWodmDJB7C4MW4mzyUV48Z6kGqPGg2JILxBTFzXf+LHmRVFDyczUEpbqAxC7H62OVdI+dzjM49DI6SB72v0H76rFnF+oYewrmHgIccCCMP8Q\/7F3JYLDzIEz1oo47pLSvDPLiMG3jY2vr+P3iY5GG\/xOjxpOB8C2\/qHwojXKs9yNl8P4fnS+7qaa1KPXYWpTYwuZ4OXurpIojx2TV4+Gg1R+JhZLV\/Uw+hwkPIepDi5YJ5A6fexlvLIKaNtzZ52F+1eNv1h57rcdpN3Vot0eeh63qc+LkeFwbsi4vX46o8aJfsXuOhatZDab6GNnaodA\/n22ifr1HXri9+rLXFtXoIfeeODVibPSh3zE3yEBx6qG2J+f2gXmwVyRSIvR5uM5+v1D+cji\/N8fBLIP534bbqqR7CTT0Uju1aTxCycxUGXKu8x3zvQj0dDDxsm7vgYXd5Ql4037vXw8mz\/yQo47ADTiavtet5T\/a9MN4alnj4aeO\/\/2Sqh8J1bDy0eAiqhzDWw\/URolk3O4qUXs0vTyOWHjLjqXgY5kFfmitemj+ZzSkDPFw8zM967ntu1+jDC8W1m2bfH3d6prTrTD54aPaQvRCsX5UY4eHiaY\/5\/dNyeE7joduYev\/DNQ+Ch+secpfglTvDQmF2w8Pqh+hLK44fnjwUL9fhwdF6Alsz6hhfklXJL9pc6f64tH6Q82sTePDloe4WViV\/yU+veIkHH+sZ42FwG+h8aFPAAx7wgAc84AEPeMADHvCABzzgAQ94wAMe8IAHPOABD3jAAx7wgAc84AEPeMADHvCABzzgAQ94wAMe8ICHZ3lYtZo+HvCAh1t5GLnd8R4GisADHir3jWcPgoc1e9910rN+mWNBjE5SRorAAx6MRUxZUHOUCDzgwVjEnBVxBoHAAx6MQcxaQ3CICDzgwVjE1CIHD3iw8jBIxKwkh3QReMCDMYh5SQ4QgQc8GIuYX+TgAQ9WHvpFTE2yt4vAAx6MQUxOsk8EHvBgLMKoyMEDHuz+sh0i5ifZ0UXgAQ\/GICySvCwCD3gwFmFZ9OMBD95FGCV5TQQe8GAMwizJKyLwgAdjEeZFPx7w4FiEZZLNIvCAB2MQtkk2isADHoxFrCn68YAHnyLMk2wRgQc8GINYkGS9CDzgwVjEwqIfD3hwJ2JNkpUi8IAHYxCrkqwSgQc8GItYXfTjAQ\/Wjc7pb6QsAg94MO4iliZZEoEHPBiLcFH04wEPTk6aVid5KgIPeDDuItYneSICD3gwFuGm6Bc84GH9SZOLJHMi8IAH4y7CSZK6CDzgwViEr6Jf8ICHpSD8JKmIwAMejEV4SvIgAg94MBbhsOgXPOBhFQhnSe5F4AEPxiK8Fjl4wMMKEf6SjEDgAQ\/GIDwmuYnAAx6MRbgucvCAB\/s25zq5x3u4ZTzRQ\/DsYWV2eHinh5rbSd+YHR7e6sF5knjAw02Koydnd8MS9yZlOHv2lr84dgEeCDzggcADHgg84IHAAx4IPOCBwAMeCDzggcADHgg84IHAAx4IPOCBwAMeCDzggcADHgg84IG4j4fum8IGt547Nyk8PKJ\/2H6J1mt8as\/q0F7FA7HCw\/fHZR4CHgg\/HiIQKzzcvDnh4bEejNoPHgjnHgQPeMDDiiaJBwIPeCDu5EHiX+zPCunfn6Ox2e3pGttjBaJXvm+JtxSP5\/5bef3QnLY3JT9ILgfBAzHbw\/ba7ufDS7u37t+2\/Vu09+YeVJNcCfm+W8IxreTr8UDM9iBBV7EnoDTSuMXHn1AFHTNJt3+2FTwQCzzsmqjsNrT3EG9UJP3G3cfzHrTtH5i4Kzrw8Nh6Oq4ADm+K64CkfYruYfeDlDwk29ffL3ggjD2E+KmUBQ+yq4klbcnRy6M8uHv8Ex6e7uE7iFPtIRSP4YP7B8aXiGke1FP0kocQj8nigXi4h5DW0GmZ\/PWw\/6NjPR166ml9K4HzJWKSB\/0CwH6AKSgDpoce4zC+pL1cPd6qejGdiouHF3qQY2EQdROSvRKnkDoU4eqliurrcSH\/OtcfiIEe1NtFt5GgZMg1mXGREbafyqF+4m++hjZilLxbTqZoMF+DGN8\/EHjAA4EHAg94IPCABwIPeCDwgAcCD3gg8IAHAg94IPCABwIPeCDwgAcCD3ggnuXh1Q0CD3jAAx7w0NAiBA8EHt7YYeABD+l28UDgIfYgeCDw8LddPBB4+G42aRPRogD3frg0HvBwabPJKhrfxTb+PcVB8EC8w4P27OrkeUJ4IN7mIeQ8zKxc8EA48xCfJh2raVn2VF88EEs9hKKHgAfi4R7Uw3\/WA+dLxGs8BDwQL\/eQPAdltwK3Uj\/ggXiPh1Dj4XltBw940Lco2oMW9++W54HAAx6iprCPvy9Sr9IdHkmHB+JBHkR0EMlDTTQbeCAeWD8068EDgYdv+SB4IPAQe6B+IPCwjbYy3krgIXmWLx6It58vPXa34gEPBB7wQOABDwQe8EDgAQ8EHvBA4AEPBB7wYNdIXhQ0LjzgAQ94wAMeHhL\/ATmhz7NSY6tCAAAAAElFTkSuQmCC","width":783}
%---
