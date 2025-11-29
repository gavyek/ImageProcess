function STGRA

clc;clear;close all;load('const_sh');

%%%%%%%%%%%%%%%%%%%%%%
% % % (1) Pre
ctGRA_bar=1;
ctGRA_bee=1;
ctGRA_stat=1;
%%%%%%%%%%%%%%%%%%%%%%

% 1 data
da=[
573135	0
158465	0
0	523043
188173	0
0	92581
0	211100
0	152862
0	0
0	0
0	211785
0	0
314430	64927
0	0
189057	48067
0	90016
36226	0
27171	143164
146993	0
503000	498171
272073	90759
43432	0
44572	227422
0	145561
275073	0
0	43719
79146	740619
886177	40253
71038	1247471
0	1453679
289044	0
0	0
0	148886
52044	40531
143489	62635
0	71286
41252	0
0	124366
133493	0
0	46778
0	0
0	0
65133	90039
0	491599
0	0
0	0
0	73628
118810	67015
43379	0
0	0
0	0
477293	0
895904	54823
0	0
0	0
0	0
0	0
0	731025
0	93045
773352	201471
39008	329745
0	60479
0	477931
0	723825
0	0
0	56216
64281	233383
42820	66296
0	0
0	0
0	328685
0	55847
306743	0
0	0
0	0
0	121478
0	0
0	274137
0	81369
0	0
0	48012
172099	0
159091	0
0	0
98838	0
199549	332161
41963	0
47794	0
669247	0
68610	0
336126	43229
0	315491
288449	101587
0	0
0	0
0	0
215434	169650
88622	39329
45104	98964
568752	425590
0	550687
1031638	nan
230364	nan
0	nan
0	nan
43393	nan
57543	nan
0	nan
0	nan
82701	nan
0	nan
0	nan
94426	nan
122462	nan
0	nan
0	nan
0	nan
1670685	nan
156812	nan
0	nan
0	nan
];
nr=size(da,1);
ne=size(da,2);
idxcmpv=1:ne-1;
% 2 file
dirfig='FIG';

% 3 parameter
% elc=cell(1,size(da,2)); % ★★★★★★★
% for k=1:size(da,2)
%     elc{k}=sprintf('e%d',k);
% end% x-axis label
elc={
'FAK-CT'
'FAK-TP'
};
swbox=0; % box option
opst='av'; % anova
% opst='kw'; % statistics option
opmt='lsd'; % post-hoc comparison option 
% opmt='hsd'; % convervative
coB=COB.navy;
coR=COR.tomato;
coG=COG.greenyellow;
coK=[0 0 0];
faco=repmat(coK, ne,1);
faap=linspace(0.1,1,ne);
thd=45; % rot angle of x axis label
Ny=30; % bee ybin
Nx=10; % bee xbin
mksz=10; % bee marker size
xystra=[1 0.5 1]; % stretch ratio
alsz=25;
atlsz=15;
atdir='out';
lnwd=2;

% (2)Process
x=1:ne;
xlm=0;
ylm=0;
% yl='FA #'; % ★★★★★★★
% yl='Area (\mmm^2)'; % ★★★★★★★
% yl='Elongation';% ★★★★★★★
yl='Intensity (a.u.)'; % ★★★★★★★
if ctGRA_bar==1
    y=da;fnm='Bar';   flefig=fullfile(dirfig, fnm);plotBarerr(x,y,'faco', faco, 'faap', faap, 'operr', 'sem', 'lnwd', 2, 'lnco', 'k', 'erlnwd', 1, 'erwdra', 0.33);axcon1('xlm', 0, 'ylm', 0, 'xl', [], 'yl', yl, 'xt', x, 'xtl', elc, 'xystra', xystra, 'thd', thd, 'optight', 'gra', 'alsz',alsz, 'atlsz', atlsz, 'atdir', atdir, 'lnwd', lnwd);
    exportgraphics(gca,sprintf('%s.jpg',flefig),'Resolution',300);hold off;close;
end
if ctGRA_bee==1
    y=da;fnm='Bee';  flefig=fullfile(dirfig, fnm);plotBeeerr(x,y,'operr', 'ci95', 'bxwdra', 0.9, 'faco', faco, 'faap', faap, 'Ny', Ny, 'erwdra', 0.25, 'egap', 0.5, 'mksz', mksz, 'rvlnwd', 1);hold on;axcon1('xlm', xlm, 'ylm', ylm, 'xl', [], 'yl', yl, 'xt', x, 'xtl', elc, 'xystra', xystra, 'alsz',alsz, 'atlsz', atlsz, 'atdir', atdir, 'lnwd', lnwd, 'thd', thd);if swbox==0;box off;end;
    exportgraphics(gca,sprintf('%s.jpg',flefig),'Resolution',300);hold off;close;
end
if ctGRA_stat ==1
    for i=1:length(idxcmpv)yl='p value';idxcmp=idxcmpv(i);
    y=da;fnm=sprintf('id#%d',idxcmp);flefig=fullfile(dirfig, fnm);plotpMCrf(x, y, idxcmp, 'dim', 1, 'opmt', opmt, 'opst', opst);axcon1('xlm', xlm, 'ylm', ylm, 'xl', [], 'yl', yl, 'xt', x, 'xtl', elc, 'xystra', xystra, 'alsz',alsz, 'atlsz', atlsz, 'atdir', atdir, 'lnwd', lnwd, 'thd', thd);if swbox==0;box off;end;
    exportgraphics(gca,sprintf('%s.jpg',flefig),'Resolution',300);hold off;close;
    end
end
end

