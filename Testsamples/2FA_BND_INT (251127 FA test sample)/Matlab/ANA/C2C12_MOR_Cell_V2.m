%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C2C12_mod_MOR_Cell_F3_V1(ctob, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Module to determine morphology of cells
% Argin
%   ## control
%   ctob:   obj switch
%   ctcal:  job switch - to calculate
%   ctplot: job switch - to plot bnd
%   ctplcp: job switch - to plot cropped bnd
%   ctxls:  job switch - to write xls
%   ## switch
%   swbd:  switch - to plot bnd {1}
%   swtx:  switch - to plot bnd # {1}
%   swscb: switch - to plot scale bar {1}
%   swval: switch - to plot value {1}
%   ## img
%   imgco:  image - fluorescence color {'g'}
%   sat: saturation level {[0.01 0.99]}
%   gam: gamma {1.0}
%   ## bnd
%   lnty:   bnd   - line type {'--'}
%   lnco:   bnd   - line color {[0.99 0.99 0.99]}
%   lnwd:   bnd   - line width {0.5}
%   ## bnd #
%   txco:   bnd # - text color {[0.99 0.99 0.99]}
%   txsz:   bnd # - text size {20}
% Author
%   Sung Sik Hur - sstahur@gmail.com
%   Last Modified - 220316

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% [1] Pre-Process
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Para#1
co_gr=[1 1 1]*0.5-eps;
% 1 default
pnm = {...
    'ctcal', 'ctplot', 'ctplcp','ctxls',  ... %%% ctl
    'swbd', 'swtx', 'swscb','swval',      ... %%% switch
    'sat', 'gam', 'imgco',                ... %%% img
    'lnty', 'lnwd', 'lnco', 'lnap',       ... %%% bnd
    'txsz', 'txco', 'txap',               ... %%% bnd #
    'bwd', 'bco', 'bap',                  ... %%% scalebar
    'flectl', 'flefle', 'fledat'          ... %%% var
    };
pdf = {...
    0,0,0,0,                              ... %%% ctl
    1, 1, 1, 1,                           ... %%% switch
    [0.01 0.99], 1.0, 'g',                ... %%% img
    '--', 0.5, co_gr, 0.5,                ... %%% bnd
    20, co_gr,1.0,                        ... %%% bnd #
    5, co_gr, 0.25,                       ... %%% scalebar
    [], [], []                            ... %%% var
    };
[ctcal, ctplot, ctplcp,ctxls,             ... %%% ctl
    swbd, swtx, swscb,swval,              ... %%% switch
    sat, gam, imgco,                      ... %%% img
    lnty, lnwd, lnco, lnap,               ... %%% bnd
    txsz, txco, txap,                     ... %%% bnd #
    bwd, bco, bap,                        ... %%% scalebar
    flectl, flefle, fledat                ... %%% var
    ] = parse_param(pnm, pdf, varargin);

% 2 control
ctall=logical(sum([ctcal, ctplot, ctplcp, ctxls]));if ctall==0; return; end %%% check control

% (2) Load
% 1 control
exc=loadfle(flectl,'mat', 'exc');

% 2 file
[fleIMGc,                                                                       ... %%% img
fleBNDmatc, fleBNDfigc, fleBNDfg1c, fleBNDfg2c, fleBNDfg3c,                     ... %%% bnd
fleBD1matc, fleBD1figc, fleBD1fg1c, fleBD1fg2c, fleBD1fg3c,                     ... %%% bnd1
fleMORmatc, fleMORfigc, fleMORfg1c, fleMORfg2c, fleMORfg3c, fleMORxls           ... %%% mor
]...
=loadfle(flefle,'mat', ...
'fleIMGc',                                                                      ... %%% img
'fleBNDmatc', 'fleBNDfigc', 'fleBNDfg1c', 'fleBNDfg2c', 'fleBNDfg3c',           ... %%% bnd
'fleBD1matc', 'fleBD1figc', 'fleBD1fg1c', 'fleBD1fg2c', 'fleBD1fg3c',           ... %%% bnd1
'fleMORmatc', 'fleMORfigc', 'fleMORfg1c', 'fleMORfg2c', 'fleMORfg3c','fleMORxls'... %%% mor
);

% 3 Variables
[imsz, imszum, imszcp, imszcpum, pxsz ...           %%% img
bsz, bszum, bszcp, bszcpum ...                      %%% scalebar
]=loadfle(fledat,'mat', ...
'imsz', 'imszum', 'imszcp', 'imszcpum', 'pxsz',...  %%% img
'bsz', 'bszum', 'bszcp', 'bszcpum' ...              %%% scalebar
);
H=imsz(1);W=imsz(2);
Hcp=imszcp(1);Wcp=imszcp(2);

% (3) Para#2
% 1 cal

% 2 plot
%%% switch
swmaxfig =1;
optight='loose';

%%% img
load('const_sh');
imgco_r = COR.tomato;
imgco_g = COG.lightgreen;
imgco_b = COB.skyblue;
satd=[0.001 0.999];
%%% mor
molnco=COB.royalblue;
molnty='-';
molnwd=2;
momksz=5;
mofaco=COB.royalblue;
txal='center';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [2] Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ii=0;
nr=size(ctob,1);
for i=1:nr
    if ctob(i)
        % (1) Parameter
        % 1 para
        ex=exc{i,1};
        fprintf('%s\n', ex);
        
        % 2 file
        %%% img
        fleim_r  = fleIMGc{i,1};
        fleim_g  = fleIMGc{i,2};
        fleim_b  = fleIMGc{i,3};
        %%% bnd
        flebndmat = fleBNDmatc{i};
        flebndfig = fleBNDfigc{i};
        flebndfg1 = fleBNDfg1c{i};
        %%% mor
        flemormat = fleMORmatc{i};
        flemorfig = fleMORfigc{i};
        flemorfg1 = fleMORfg1c{i};
        
        % (2) Load
        %%% img
        im_r=imreadfle(fleim_r,'tif');imaj_r=imad(im_r,sat, gam);imco_r=imgray2rgb(imaj_r,imgco_r);
        im_g=imreadfle(fleim_g,'tif');imaj_g=imad(im_g,sat, gam);imco_g=imgray2rgb(imaj_g,imgco_g);
        im_b=imreadfle(fleim_b,'tif');imaj_b=imad(im_b,sat, gam);imco_b=imgray2rgb(imaj_b,imgco_b);
        imco_rgb=cat(3, imaj_r, imaj_g, imaj_b);

        %%% bnd
        % bdc
        [tbdc, rbdc, thsv]=loadfle(flebndmat,'mat', 'tbdc', 'rbdc', 'thsv');
        bdc=tbdc;
        % bwc
        [~, bwc]=bd2bwmaxc(bdc, imsz);
        % bwc all
        bwa=addBWc(bwc);
        % bdumc
        bdumc = cvbd_px2umc(bdc, imsz, pxsz);
        
        %%% rect
        [xlmc,ylmc, xlmumc, ylmumc]=recbdc(bdc, imszcp, imsz, pxsz);
        nbd=length(bdc);
        
        % (3) Cal
        % 1 cal
        if ctcal==1
            [ctc,  arv,  lnav,  lnbv,  elv, crv, orv, rfv, ctumc,arumv,lnaumv,lnbumv] = calmobdc(bdc, imsz, pxsz);
            savefle(flemormat,'mat', ctc,  arv,  lnav,  lnbv,  elv, crv, orv, rfv, ctumc,arumv,lnaumv,lnbumv);
        end
        if ctcal==0
            [ctc,  arv,  lnav,  lnbv,  elv, crv, orv, rfv, ctumc,arumv,lnaumv,lnbumv] = loadfle(flemormat,'mat', ...
                'ctc',  'arv',  'lnav',  'lnbv',  'elv', 'crv', 'orv', 'rfv', 'ctumc','arumv','lnaumv','lnbumv');
        end
        % 2 data in bd
        [imbdc_r, ~] = dapoc_mt(bwc, 0, 'mean', 90, imaj_r); % int 
        
        % (4) Plot
        if ctplot==1
            %%% im
            imco=imco_r;

            %%% plot
            imshow(imco);hold on;
            for k=1:nbd
                %%% bd
                bd=bdc{k};
                %%% line
                ct=ctc{k};
                or=orv(k);
                xy=getln_thd(ct, -or, [1 1 W H], 'num', 500);
                xyb =ptbd(xy, bd, imsz);
                plotbd(xyb, 'lnco',  molnco, 'lnty', molnty, 'lnwd', molnwd, 'mk', 'none');
                plot(ct(1),ct(2), 'Marker', 'o', 'MarkerSize', momksz, 'MarkerEdgeColor', 'none', 'MarkerFaceColor', mofaco); % centroid
            end
            if swbd==1;plotbdc(bdc, 'lnty',lnty, 'lnwd', lnwd, 'lnco', [lnco, lnap], 'swtx', swtx, 'txsz', txsz, 'txco', [txco, txap]);end
            if swscb==1;addscb([W*(1-0.03)-bsz, H*(1-0.05)], bsz, 'bco', [bco, bap], 'bwd', bwd);end; axis off;
            axcon1('xlm', [0 W], 'ylm', [0 H], 'xt', [], 'yt', [], 'optight', optight);box on;
            exportgraphics(gca,sprintf('%s.jpg',flemorfig),'Resolution',300);close;
        end
        
        % (5) Plot crop
        if ctplcp==1
            for k=1:nbd
                %%% img
                im=uint16(imbdc_r{k});
                imgco=imgco_r;
                
                imco=imgray2rgb(im,imgco);
                %%% bd
                bd = bdc{k};
                bw = bwc{k};
                %%% crop
                xlm=xlmc{k};
                ylm=ylmc{k};
                
                %%% plot
                imshow(imco);hold on;
                if swbd==1;plotbdc(bd, 'lnty',lnty, 'lnwd', lnwd, 'lnco', [lnco, lnap], 'swtx', 0, 'txsz', txsz, 'txco', [txco, txap]);end
                if swscb==1;addscb([xlm(2)-bszcp-Wcp*0.05, ylm(2)-Hcp*0.05], bszcp, 'bco', [bco, bap], 'bwd', bwd);end
                if swval;textxy(mean(bd,1), sprintf('AR#%d\n%0.2g',k, arumv(k)), 'txco', txco, 'txsz', txsz, 'txal', txal);end
                if swscb==1;addscb([xlm(2)-bszcp-Wcp*0.05, ylm(2)-Hcp*0.05], bszcp, 'bco', [1 1 1 0.5]-eps, 'bwd', 7);end
                axcon1('xlm', xlm, 'ylm', ylm, 'xt', [], 'yt', [], 'optight', optight);
                exportgraphics(gca,sprintf('%s-%d.jpg',flemorfg1, k),'Resolution',300);hold off;close;
            end
        end
        
        % (6) xls
        if ctxls==1
            for k=1:nbd
                ii=ii+1;
                vex{ii,1} = exc{i,1};
                vep(ii,1) = exc{i,3};
                vsp(ii,1) = exc{i,4};
                vcp(ii,1) = k;
                
                var(ii,1) = arumv(k);
                vln(ii,1) = lnaumv(k);
                vel(ii,1) = elv(k);
                vcr(ii,1) = crv(k);
                vrf(ii,1) = rfv(k);
            end
        end
    end
end
if ctxls==1
    % (5) xls list
    % 1 parameter
    flexls=fleMORxls;
    %ne=max(vep(:));
    clear H1 H2 H3 B1 B2 B3 res;
    
    % 2 head
    H1={'ex'};
    H2={'ep', 'sp','cp'};
    H3={'ar', 'ln', 'el', 'cr', 'rf'};
    
    % 3 body
    B1=vex;
    B2=[vep, vsp, vcp];B2=num2cell(B2);
    B3=[var, vln, vel, vcr, vrf];B3=num2cell(B3);
    
    % 4 write
    res=[H1, H2, H3;B1, B2, B3];
    sht='ALL list';saveflexls(flexls, 'xlsx', sht, res);
    
    % (6) xls table
    % 0 parameter
    clear H1 H2 H3 B1 B2 B3 res;
    gv=vep;
    gi=1:max(vep(:));n1=length(gi);
    % 1 data
    [mep,msp,mcp, mar,mln,mel,mcr, mrf]=v2m_1Dmt(gv, gi,        vep,vsp,vcp, var,vln,vel,vcr, vrf);
    [mep,msp,mcp, mar,mln,mel,mcr, mrf]=transposemt(              mep,msp,mcp, mar,mln,mel,mcr, mrf);
    iempty=logical(prod(isnan(mep),1)); % empty col index
    [mep,msp,mcp, mar,mln,mel,mcr, mrf]=rmcolmt(iempty,          mep,msp,mcp, mar,mln,mel,mcr, mrf);
    
    % 2 head
    nr=size(mep,1);
    nc=size(mep,2);
    [H1, H3]=deal(cell(1,nc));
    H2={'n'};
    for j=1:nc
        if ~isnan(mep(1,j))
            H1{1,j}=sprintf('e%d', mep(1,j));
            H3{1,j}=sprintf('e%d', mep(1,j));
        end
    end
    
    % 3 Body
    B1=cell(nr,nc);
    B2=cell(nr,1);
    for i=1:nr
        for j=1:nc
            if ~isnan(mep(i,j))
                B1{i,j}=sprintf('e%ds%dc%d',mep(i,j),msp(i,j), mcp(i,j));
            end
        end
    end
    for i=1:nr
        B2{i,1}=sprintf('n%d', i);
    end
    
    % 4. Write
    B3=num2cell(mep);sht='ep';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(msp);sht='sp';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mcp);sht='cp';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mar);sht='ar';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mln);sht='ln';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mel);sht='el';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mcr);sht='cr';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mrf);sht='rf';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
end
end