%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C2C12_mod_INT_Cell_F3_V1(ctob, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Module Code - to determine intensity of cell
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
%   ## plot
%   cmaxv:  color limit - {0}
% Author
%   Sung Sik Hur - sstahur@gmail.com
%   Last Modified - 220323

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
    'cmaxv',                              ... %%% plot
    'flectl', 'flefle', 'fledat'          ... %%% var
    };
pdf = {...
    0,0,0,0,                              ... %%% ctl
    1, 1, 1, 1,                           ... %%% switch
    [0.01 0.99], 1.0, 'g',                ... %%% img
    '--', 0.5, co_gr, 0.5,                ... %%% bnd
    20, co_gr,1.0,                        ... %%% bnd #
    5, co_gr, 0.25,                       ... %%% scalebar
    0,                                    ... %%% plot  
    [], [], []                            ... %%% var
    };
[ctcal, ctplot, ctplcp,ctxls,             ... %%% ctl
    swbd, swtx, swscb,swval,              ... %%% switch
    sat, gam, imgco,                      ... %%% img
    lnty, lnwd, lnco, lnap,               ... %%% bnd
    txsz, txco, txap,                     ... %%% bnd #
    bwd, bco, bap,                        ... %%% scalebar
    cmaxv,                                ... %%% plot
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
fleMORmatc, fleMORfigc, fleMORfg1c, fleMORfg2c, fleMORfg3c, fleMORxls,          ... %%% mor
fleMO1matc, fleMO1figc, fleMO1fg1c, fleMO1fg2c, fleMO1fg3c, fleMO1xls,          ... %%% mor1
fleINTmatc, fleINTfigc, fleINTfg1c, fleINTfg2c, fleINTfg3c, fleINTxls,          ... %%% int
fleIT1matc, fleIT1figc, fleIT1fg1c, fleIT1fg2c, fleIT1fg3c, fleIT1xls           ... %%% it1
]...
=loadfle(flefle,'mat', ...
'fleIMGc',                                                                          ... %%% img
'fleBNDmatc', 'fleBNDfigc', 'fleBNDfg1c', 'fleBNDfg2c', 'fleBNDfg3c',               ... %%% bnd
'fleBD1matc', 'fleBD1figc', 'fleBD1fg1c', 'fleBD1fg2c', 'fleBD1fg3c',               ... %%% bnd1
'fleMORmatc', 'fleMORfigc', 'fleMORfg1c', 'fleMORfg2c', 'fleMORfg3c','fleMORxls',   ... %%% mor
'fleMO1matc', 'fleMO1figc', 'fleMO1fg1c', 'fleMO1fg2c', 'fleMO1fg3c', 'fleMO1xls',  ... %%% mor1
'fleINTmatc', 'fleINTfigc', 'fleINTfg1c', 'fleINTfg2c', 'fleINTfg3c', 'fleINTxls',  ... %%% int
'fleIT1matc', 'fleIT1figc', 'fleIT1fg1c', 'fleIT1fg2c', 'fleIT1fg3c', 'fleIT1xls'   ... %%% it1
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
bgra=0.01;

% 2 plot
%%% switch
swmaxfig =1;
swcb=1;
optight='loose';

%%% img
load('const_sh');
imgco_r = COR.tomato;
imgco_g = COG.lightgreen;
imgco_b = COB.skyblue;
satd=[0.001 0.999];

%%% int
cmap=parula;
txal='center';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [2] Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ii=0;
jj=0;
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
        flebd1mat = fleBD1matc{i};
        %%% mor
        flemormat = fleMORmatc{i};
        flemo1mat = fleMO1matc{i};
        %%% int
        fleintmat = fleINTmatc{i};
        fleintfig = fleINTfigc{i};
        fleintfg1 = fleINTfg1c{i};
        %%% para
        cmax=cmaxv(i); % color
                
        % (2) Load
        %%% img
        im_r=imreadfle(fleim_r,'tif');imaj_r=imad(im_r,sat, gam);imco_r=imgray2rgb(imaj_r,imgco_r);
        im_g=imreadfle(fleim_g,'tif');imaj_g=imad(im_g,sat, gam);imco_g=imgray2rgb(imaj_g,imgco_g);
        im_b=imreadfle(fleim_b,'tif');imaj_b=imad(im_b,sat, gam);imco_b=imgray2rgb(imaj_b,imgco_b);
        imco_rgb=cat(3, imaj_r, imaj_g, imaj_b);

        %%% bnd 
        [tbdc, rbdc, thsv]=loadfle(flebndmat,'mat', 'tbdc', 'rbdc', 'thsv');
        % bdc
        bdc=tbdc;
        % bwc
        [~, bwc]=bd2bwmaxc(bdc, imsz);
        % bwa
        bwa=addBWc(bwc);
        nbd=length(bdc);
        
        % (3) Cal
        if ctcal==0
            [ita, itmv, itsv, bg]=loadfle(fleintmat,'mat', 'ita', 'itmv', 'itsv', 'bg');
        end
        if ctcal==1
            im=im_r;
            
            bg =imBGhist1(im, 'bgra', bgra);
            im = im-bg;

            % 2 int map and mean
            [ita,~]    = dapo_mt(bwa,  0, 'mean', 90, im); % int nuc
            [itc,itmv] = dapoc_mt(bwc, 0, 'mean', 90, im); % int nuc (mean)
            itsv=nan(1,nbd);
            for k=1:nbd
                itsv(k)=sum(itc{k}(:), 'omitnan');
            end
            
            % 3 save
            savefle(fleintmat,'mat', ita, itmv, itsv, bg);
        end
        
        % (4) Plot
        if cmax==0;clm=0;else;clm=[0 cmax];end
        if ctplot==1
            %%% im
            im=ita;
            
            imshowmap(im, 'cmap', cmap, 'swcb', swcb, 'clm', clm);hold on;axis tight;
            %%% bd
            if swbd==1;plotbdc(bdc, 'lnty',lnty, 'lnwd', lnwd, 'lnco', [lnco, lnap], 'swtx', 0, 'txsz', txsz, 'txco', [txco, txap]);end
            if swscb==1;addscb([W*(1-0.03)-bsz, H*(1-0.05)], bsz, 'bco', [bco, bap], 'bwd', bwd);end; axis off;
            %if swval==1;textxy(round([W*0.5, H*0.5]), sprintf('ar\n%0.0f um^2',sum(itsv, 'all', 'omitnan')), 'txco', txco, 'txsz', txsz, 'txal', txal);end
            axcon1('xlm', [0 W], 'ylm', [0 H], 'xt', [], 'yt', [], 'optight', optight);box on;
            exportgraphics(gca,sprintf('%s.jpg',fleintfig),'Resolution',300);hold off;close;
        end
        
        % (5) plot crop
        if ctplcp==1
            for k=1:nbd
                im=ita;
                
                bd=bdc{k};
                bw = bwc{k};
                imbd = getimbw(im, bw);
                imbd = double(imbd);
                [xlm,ylm, xlmum, ylmum]=recbd(bd, imszcp, imsz, pxsz);
                imshowmap(imbd, 'cmap', cmap, 'swcb', swcb, 'clm', clm);hold on;axis tight;
                plotbdc(bd, 'lnty',lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd, 'swtxt', 0, 'txsz', txsz, 'txco', txco);
                if swval;textxy(mean(bd), sprintf('it#%d\n%0.2g',k, itsv(k)), 'txco', txco, 'txsz', txsz, 'txal', txal);end
                if swscb==1;addscb([xlm(2)-bszcp-Wcp*0.05, ylm(2)-Hcp*0.05], bszcp, 'bco', [1 1 1 0.5]-eps, 'bwd', 7);end
                axcon1('xlm', xlm, 'ylm', ylm, 'xt', [], 'yt', [], 'optight', optight);
                exportgraphics(gca,sprintf('%s-%d.jpg',fleintfg1, k),'Resolution',300);hold off;close;
            end
        end
        
        
        
        
        % (6) xls
        if ctxls==1
            for k=1:nbd
                ii=ii+1;
                vex{ii,1} = sprintf('e%ds%dc%d', exc{i,3}, exc{i,4}, k);
                vep(ii,1) = exc{i,3};
                vsp(ii,1) = exc{i,4};
                vcp(ii,1) = k;
                vitm(ii,1) = itmv(k);
                vits(ii,1) = itsv(k);
            end
        end
    end
end
if ctxls==1
    % (5) xls list
    % 1 parameter
    flexls=fleINTxls;
    clear H1 H2 H3 B1 B2 B3 res;
    
    % 2 head
    H1={'ex'};
    H2={'ep', 'sp','cp'};
    H3={'itm', 'its'};
    
    % 3 body
    B1=vex;
    B2=[vep, vsp, vcp];B2=num2cell(B2);
    B3=[vitm, vits];B3=num2cell(B3);
    
    % 4 write
    res=[H1, H2, H3;B1, B2, B3];
    sht='ALL list';saveflexls(flexls, 'xlsx', sht, res);
    
    % (6) xls table
    % 0 parameter
    clear H1 H2 H3 B1 B2 B3 res;
    gv=vep;
    gi=1:max(vep(:));n1=length(gi);
    
    % 1 data
    [mep,msp,mcp, mitm,mits]=v2m_1Dmt(gv, gi, vep,vsp,vcp, vitm, vits);
    [mep,msp,mcp, mitm,mits]=transposemt(             mep,msp,mcp, mitm,mits);
    iempty=logical(prod(isnan(mep),1)); % empty col index
    [mep,msp,mcp, mitm,mits]=rmcolmt(iempty,          mep,msp,mcp, mitm,mits);

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
    
    % 4 Write
    B3=num2cell(mep);sht='ep';  
    res=[H1,H2 H3;B1,B2,B3];
    saveflexls(flexls, 'xlsx', sht, res);
    
    B3=num2cell(msp);sht='sp';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mcp);sht='cp';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mitm);sht='itm';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mits);sht='its';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
end
end