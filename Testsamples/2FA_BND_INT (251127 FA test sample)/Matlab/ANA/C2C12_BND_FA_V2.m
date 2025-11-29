%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C2C12_mod_BND_FA_F3_V1(ctob, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Module Code - to determine bnd of FAs with absolute threshold value
% Argin
%   ## control
%   ctob:   obj switch
%   ctcal:  job switch      - to calculate
%   ctplot: job switch job  - to plot bnd
%   ctplcp: job switch job  - to plot cropped bnd
%   ## switch
%   swbd:   switch - to plot bnd {1}
%   swtxt:  switch - to plot bnd # {1}
%   swscb:  switch - to plot scale bar {1}
%   opcal:  option switch - {'n'}==new, 'a'==add, 'r'==revise, 'd'==delete
%   ## img
%   imgco:  image - fluorescence color {'g'}
%   sat: saturation level {[0.01 0.99]}
%   gam: gamma {1.0}
%   ## bnd
%   lnty:   bnd   - line type {'--'}
%   lnwd:   bnd   - line width {0.5}
%   lnco:   bnd   - line color {[0.99 0.99 0.99]}
%   lnap:   bnd   - line alpha {0.5}
%   ## bnd #
%   txco:   bnd # - text color {[0.99 0.99 0.99]}
%   txsz:   bnd # - text size {20}
%   txap:   bnd # - text alpha {0.5}
% Author
%   Sung Sik Hur - sstahur@gmail.com
%   Last Modified - 220408

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [1] PRE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Def
pnm = {...
    'ctcal', 'ctplot', 'ctplcp',        ... %%% control
    'swbd', 'swtx', 'swscb',            ... %%% switch
    'opcal', 'opths',                   ... %%% option
    'ths','alpha',                      ... %%% cal img
    'sat', 'gam', 'imgco',              ... %%% img
    'lnty', 'lnwd', 'lnco', 'lnap',     ... %%% bnd
    'txsz', 'txco', 'txap',             ... %%% bnd #
    'bwd', 'bco', 'bap',                ... %%% scalebar
    'flectl', 'flefle', 'fledat'        ... %%% var
    };
pdf = {...
    0,0,0,                              ... %%% control
    1, 1, 1,                            ... %%% switch
    'n', 'int',                         ... %%% option
    0.04, 2,                            ... %%% cal img
    [0.01 0.99], 1.0, 'g',              ... %%% img
    '--', 0.5, [1 1 1]-eps, 0.5,        ... %%% bnd
    20, [1 1 1]-eps,1.0,                ... %%% bnd #
    5, [1 1 1]-eps, 0.25,               ... %%% scalebar
    [], [], []                          ... %%% var
    };
[ctcal, ctplot, ctplcp,                 ... %%% control
    swbd, swtx, swscb,                  ... %%% switch
    opcal, opths,                       ... %%% option
    ths,alpha                                ... %%% cal img
    sat, gam, imgco,                    ... %%% img
    lnty, lnwd, lnco, lnap,             ... %%% bnd
    txsz, txco, txap,                   ... %%% bnd #
    bwd, bco, bap,                      ... %%% scalebar
    flectl, flefle, fledat              ... %%% var
    ] = parse_param(pnm, pdf, varargin);

% (2) Control
ctall=logical(sum([ctcal, ctplot, ctplcp]));if ctall==0; return; end        %%% check

% (3) Load
% 1 control
exc=loadfle(flectl,'mat', 'exc'); % load exc

% 2 file
[fleIMGc,                                                               ... %%% img
fleBNDmatc, fleBNDfigc, fleBNDfg1c, fleBNDfg2c, fleBNDfg3c,             ... %%% bnd
fleBD2matc, fleBD2figc, fleBD2fg1c, fleBD2fg2c, fleBD2fg3c              ... %%% bnd3
]...
=loadfle(flefle,'mat', ...
'fleIMGc',                                                              ... %%% img
'fleBNDmatc', 'fleBNDfigc', 'fleBNDfg1c', 'fleBNDfg2c', 'fleBNDfg3c',   ... %%% bnd
'fleBD2matc', 'fleBD2figc', 'fleBD2fg1c', 'fleBD2fg2c', 'fleBD2fg3c'    ... %%% bnd3
);

% 3 var
[imsz, imszum, imszcp, imszcpum, pxsz               ... %%% img
    bsz, bszum, bszcp, bszcpum                      ... %%% scalebar
    ]=loadfle(fledat,'mat', ...
    'imsz', 'imszum', 'imszcp', 'imszcpum', 'pxsz', ... %%% img
    'bsz', 'bszum', 'bszcp', 'bszcpum',             ... %%% scalebar
    'arum_sm', 'arum_lg', 'ar_sm', 'ar_lg'          ... %%% FA size filter
    );
H=imsz(1);W=imsz(2);
Hcp=imszcp(1);Wcp=imszcp(2);

% (4) Parameters
% 1 cal
%%% switch
swopen = 1;     % switch for oepn filter - to remove small holes {0}
swclose = 0;    % switch for close filter - to remove small protursions {0}
swszsm = 1;     % swicth for small size filter {1}
swszlg = 1;     % swicth for large size filter {1}
%%% imp
nh_open = 2; % in px
nh_close = 2; % in px
arum_sm=1.5; % minimum area in um^2 ★
arum_lg=30; % maximum area in um^2 ★
ar_sm= round(arum_sm /pxsz(1)/pxsz(2));
ar_lg= round(arum_lg /pxsz(1)/pxsz(2));

% 2 plot
%%% switch
swmaxfig=0;   % switch to maximize window: {1}
optight='loose';
ps_qs=[0.1, 0.8];% cell quest box location

%%% img
load('const_sh');
imgco_r=COR.tomato;
imgco_g=COG.lawngreen;
imgco_b=COB.skyblue;
faap=0.2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [2] PRO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        flebndmat = fleBNDmatc{i}; % Cell
        flebd2mat = fleBD2matc{i}; % FA
        flebd2fig = fleBD2figc{i};
        flebd2fg1 = fleBD2fg1c{i};
        
        % (2) Load
        %%% img
        im_r=imreadfle(fleim_r,'tif');imaj_r=imad(im_r,sat, gam);imco_r=imgray2rgb(imaj_r,imgco_r);
        im_g=imreadfle(fleim_g,'tif');imaj_g=imad(im_g,sat, gam);imco_g=imgray2rgb(imaj_g,imgco_g);
        im_b=imreadfle(fleim_b,'tif');imaj_b=imad(im_b,sat, gam);imco_b=imgray2rgb(imaj_b,imgco_b);
        
        %%% bnd - cell
        [tbdc, rbdc, thsv]=loadfle(flebndmat,'mat', 'tbdc', 'rbdc', 'thsv'); % Cell
        bdc_cell = tbdc;
        nbd=length(tbdc);
        
        % (3) Cal
        if ctcal==0
            [bdokcc, bwokc, stokc,          ... %%% FA ok
                rbdc, thsv,alphav,          ... %%% rbd & ths
                bdlgcc, bwlgc, stlgc,       ... %%% FA large
                bdsmcc, bwsmc, stsmc        ... %%% FA small
            ]=loadfle(flebd2mat,'mat',      ...
                'bdokcc', 'bwokc', 'stokc', ... %%% FA ok
                'rbdc', 'thsv','alphav',    ... %%% rbd & ths
                'bdlgcc', 'bwlgc', 'stlgc',  ... %%% FA large
                'bdsmcc', 'bwsmc', 'stsmc'   ... %%% FA small
                );
        end
        if ctcal==1
            im=im_g;
            [bdokcc, bwokc, stokc, ... %%% FA ok
            rbdc, thsv,alphav,     ... %%% rbd & ths
            bdlgcc, bwlgc, stlgc,  ... %%% FA large
            bdsmcc, bwsmc, stsmc   ... %%% FA small
            ] = fathssz2_nc(im,...
            'swrbdo', 1,'swmaxfig', swmaxfig, 'swbd', swbd,                                     ... %%% switch
            'swszsm', swszsm, 'swszlg', swszlg, 'swopen', swopen, 'swclose', swclose,           ... %%% switch filters
            'opths', opths, ...                                                                 ... %%% option
            'bitdepth', 16, 'ar_sm',ar_sm,'ar_lg',ar_lg,'nh_open',nh_open,'nh_close',nh_close,  ... %%% cal img
            'rbdoc', bdc_cell, 'ths', ths,'alpha', alpha,                                       ... %%% cal bnd
            'sat', sat, 'gam', gam, 'imgco', imgco,                                             ... %%% plot img
            'lnwd', lnwd, 'lnco', lnco, 'faap', faap,                                           ... %%% plot bnd
            'tl', ex                                                                            ... %%% plot etc
             );
            savefle(flebd2mat,'mat',  ...
                bdokcc, bwokc, stokc, ... %%% FA ok
                rbdc, thsv,alphav,    ... %%% rbd & ths
                bdlgcc, bwlgc, stlgc,  ... %%% FA large
                bdsmcc, bwsmc, stsmc   ... %%% FA small
            );
        end
        bdcc_FA = bdokcc;
        bwc_FA  = bwokc;
        
        % (4) Plot
        if ctplot==1
            %%% img
            im=imaj_g;
            imgco=imgco_g;
            %%% bd
            bd1c  = bdc_cell;
            bd2cc = bdcc_FA;
            bw2c  = bwc_FA;
            bw2   = addBWc(bwc_FA);
            %%% im bd
            imbd = getimbw(im, bw2);
            imco=imgray2rgb(imbd,imgco);
            
            %%% plot
            imshow(imco, 'Border', 'tight');hold on;
            plotbdc(bd1c, 'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd, 'swtx', swtx, 'txco', lnco);
            if swbd==1;for k=1:nbd;plotbdc(bd2cc{k}, 'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd);end;end
            if swscb;addscb([W*(1-0.03)-bsz, H*(1-0.05)], bsz, 'bco', [bco, bap], 'bwd', bwd);end
            exportgraphics(gca,sprintf('%s.jpg',flebd2fig),'Resolution',300);hold off;close;
        end
        
        % (5) Plot crop
        if ctplcp==1
            %%% img
            im=imaj_g;
            imgco=imgco_g;
            %%% bd
            bd1c  = bdc_cell;
            bd2cc = bdcc_FA;
            bw2c  = bwc_FA;
            bw2   = addBWc(bwc_FA);
            %%% im rect
            [xlmc, ylmc, xlmumc, ylmumc]=recbdc(bd1c, imszcp, imsz, pxsz);
            
            for k=1:nbd
                %%% bd
                bd1 = bd1c{k};
                bd2c= bd2cc{k}; 
                bw2 = bw2c{k};
                %%% rec
                xlm = xlmc{k};
                ylm = ylmc{k};
                %%% im bd
                imbd = getimbw(im, bw2);
                imco=imgray2rgb(imbd,imgco);
                
                %%% plot
                imshow(imco, 'Border', 'tight');hold on;
                if swbd==1;plotbdc(bd1, 'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd);end
                plotbdc(bd1, 'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd);
                if swbd==1;plotbdc(bd2c, 'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd);end;
                if swscb==1;addscb([xlm(2)-bszcp-Wcp*0.05, ylm(2)-Hcp*0.05], bszcp, 'bco', [1 1 1 0.5]-eps, 'bwd', 7);end
                axcon1('xlm', xlm, 'ylm', ylm, 'xt', [], 'yt', [], 'optight', optight);
                exportgraphics(gca,sprintf('%s-%d.jpg',flebd2fg1, k),'Resolution',300);hold off;close;
            end
        end
    end
end





