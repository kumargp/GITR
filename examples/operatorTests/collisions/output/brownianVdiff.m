
nP = 1e4;
x = 5e3*ones(1,nP);
y = zeros(1,nP);
z = zeros(1,nP);
dt = 1e-6;

T=20;
m1 = 184*1.66e-27;
k = 1.38e-23*11604;% m2 kg s-2 eV-1
sigma = sqrt(k*T/m1); %meters per second
nu = 1e2;
D = sigma^2*nu;
nu_E = 1e3;
nu_s0 = 500;
D_E = sigma^2*nu_E;
nT = 1e4;
stepSize = sqrt(D*dt);
stepSize_E = sqrt(D_E*dt);
B_unit = [0,0,1];
tic
for i=1:nT
    vPartNorm = sqrt(x.^2 + y.^2 + z.^2);
    vxRelative = x;% - flowVx;
    vyRelative = y;% - flowVy;
    vzRelative = z;% - flowVz;
    vRel2 = vxRelative.^2 + vyRelative.^2 + vzRelative.^2;
    E = 0.5*m1/1.602e-19*vRel2;
    vRelativeNorm = sqrt(vRel2);
    
    d_parx = vxRelative./vRelativeNorm;
    d_pary = vyRelative./vRelativeNorm;
    d_parz = vzRelative./vRelativeNorm;
    factor0 = B_unit(1)*d_parx + B_unit(2)*d_pary + B_unit(3)*d_parz;
    thisInd = find(abs(factor0-1) < 1e-5);
        factor1 = 1./sqrt(1-factor0.^2);
        e1x = factor1 .* ((factor0.*d_parx) - B_unit(1));
        e1y = factor1 .* ((factor0.*d_pary) - B_unit(2));
        e1z = factor1 .* ((factor0.*d_parz) - B_unit(3));
        factor1 = -1*factor1;
        e2x = factor1 .* (B_unit(3)*d_pary - B_unit(2)*d_parz);
        e2y = factor1 .* (B_unit(1)*d_parz - B_unit(3)*d_parx);
        e2z = factor1 .* (B_unit(2)*d_parx - B_unit(1)*d_pary);  

        d_parx(thisInd) = 0.001;
        d_pary(thisInd) = 0.001;
        d_parz(thisInd) = 1;
        dparMag = sqrt(d_parz(thisInd).^2 + d_pary(thisInd).^2 + d_parz(thisInd).^2);
        d_parx(thisInd) = d_parx(thisInd)./dparMag;
        d_pary(thisInd) = d_pary(thisInd)./dparMag;
        d_parz(thisInd) = d_parz(thisInd)./dparMag;
        cn_theta = d_parz(thisInd);
        sn_theta = sqrt(d_parx(thisInd).^2 + d_pary(thisInd).^2);
        st0ind = find(sn_theta == 0);
        sn_theta(st0ind) = 0.001;
        sn_phi = d_pary(thisInd)./sn_theta;
        cn_phi = d_parx(thisInd)./sn_theta;
        e1x(thisInd) = cn_theta.*cn_phi;
        e1y(thisInd) = cn_theta.*sn_phi;
        e1z(thisInd) = -sn_theta;
        e2x(thisInd) = -sn_phi;
        e2y(thisInd) = cn_phi;
        e2z(thisInd) = 0;
        
    n1 = normrnd(0,1,1,nP);    
    r1 = rand(1,nP);
    r2 = rand(1,nP);
        r3 = rand(1,nP);
        r4 = rand(1,nP);
        r5 = rand(1,nP);
    plumin1 = 2*floor(r1 + 0.5)-1;
    plumin2 = 2*floor(r2 + 0.5)-1;
    pluminE = 2*floor(r3 + 0.5)-1;
    pluminE2 = 2*floor(r4 + 0.5)-1;
    pluminE3 = 2*floor(r5 + 0.5)-1;
    drift = -dt*nu_s0.*vRelativeNorm;
    coeff_par = pluminE.*stepSize_E;
    coeff_par2 = pluminE.*stepSize_E;
    coeff_par3 = pluminE.*stepSize_E;
    coeff_perp1 = stepSize*plumin1;
    coeff_perp2 = stepSize*plumin2;
    dx = coeff_par.*d_parx + coeff_perp1.*e1x + coeff_perp2.*e2x;
    dy = coeff_par.*d_pary + coeff_perp1.*e1y + coeff_perp2.*e2y;
    dz = coeff_par.*d_parz + coeff_perp1.*e1z + coeff_perp2.*e2z;
    x = x + dx;
    y = y + dy;
    z = z + dz;
    vPartNorm = sqrt(x.^2 + y.^2 + z.^2);
    x = (vRelativeNorm + drift.*d_parx + coeff_par.*d_parx).*x./vPartNorm;
    y = (vRelativeNorm + drift.*d_pary + coeff_par.*d_pary).*y./vPartNorm;
    z = (vRelativeNorm + drift.*d_parz + coeff_par.*d_parz).*z./vPartNorm;
%       x = x + coeff_par;
%       y = y + coeff_par2;
%       z = z + coeff_par3;
end
toc
figure(1)
h1 = histogram(x)
hold on
xgrid = linspace(-max(abs(x)),max(abs(x)));
h1 = histogram(y)
histogram(z)
% sigma = sigma;
mu = 0;
f = 1/sqrt(2*pi*sigma*sigma)*exp(-(xgrid-mu).^2/(2*sigma.^2));
plot(xgrid,f*max(h1.Values)/max(f))

Egrid = linspace(0,200)*1.602e-19;
fe = 2*sqrt(Egrid./pi).*(1/(k*T))^(1.5).*exp(-Egrid./(k*T));
figure(3)
h2=histogram(E)
hold on
plot(Egrid/1.602e-19,fe*max(h2.Values)/max(fe))