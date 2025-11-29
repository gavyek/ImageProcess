%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C2C12_BND_Cell_V2(ctob, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Module Code - to determine bnd of cells
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
%   Last Modified - 230116

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [1] Pre-Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Parameter#1
% 1 default
pnm = {...
'ctcal', 'ctplot', 'ctplcp',            ... %%% control
'swbd', 'swtx', 'swscb', 'opcal',       ... %%% switch
'swszsm', 'swszlg', 'swopen', 'swclose',... %%% switch filters
'sat', 'gam', 'imgco',                  ... %%% img
'lnty', 'lnwd', 'lnco', 'lnap',         ... %%% bnd
'ar_open','ar_close',                   ... %%% bnd cal
'txsz', 'txco', 'txap',                 ... %%% bnd #
'bwd', 'bco', 'bap',                    ... %%% scalebar
'flectl', 'flefle', 'fledat'            ... %%% var
};
pdf = {...
0,0,0,                                  ... %%% control
1, 1, 1, 'n',                           ... %%% switch
1,1,1,1,                                ... %%% switch filters
[0.01 0.99], 1.0, 'g',                  ... %%% img
'--', 0.5, [1 1 1]-eps, 0.5,            ... %%% bnd
500, 10,                                ... %%% bnd cal
20, [1 1 1]-eps,1.0,                    ... %%% bnd #
5, [1 1 1]-eps, 0.25,                   ... %%% scalebar
[], [], []                              ... %%% var
};
[ctcal, ctplot, ctplcp,                 ... %%% control
 swbd, swtx, swscb, opcal,              ... %%% switch
 swszsm, swszlg, swopen, swclose,       ... %%% switch filters
 sat, gam, imgco,                       ... %%% img
 lnty, lnwd, lnco, lnap,                ... %%% bnd
 ar_open,ar_close,                      ... %%% bnd cal
 txsz, txco, txap,                      ... %%% bnd #
 bwd, bco, bap,                         ... %%% scalebar
 flectl, flefle, fledat                 ... %%% var
] = parse_param(pnm, pdf, varargin);

% 2 control
ctall=logical(sum([ctcal, ctplot, ctplcp]));if ctall==0; return; end %%% check control

% (2) Load
% 1 control
exc=loadfle(flectl,'mat', 'exc');

% 2 file
[fleIMGc,                                                                   ... %%% img
    fleBNDmatc, fleBNDfigc, fleBNDfg1c, fleBNDfg2c, fleBNDfg3c              ... %%% bnd
    ]...
    =loadfle(flefle,'mat', ...
    'fleIMGc',                                                              ... %%% img
    'fleBNDmatc', 'fleBNDfigc', 'fleBNDfg1c', 'fleBNDfg2c', 'fleBNDfg3c'    ... %%% bnd
    );

% 3 Variables
[imsz, imszum, imszcp, imszcpum, pxsz               ... %%% img
    bsz, bszum, bszcp, bszcpum                      ... %%% scalebar
    ]=loadfle(fledat,'mat', ...
    'imsz', 'imszum', 'imszcp', 'imszcpum', 'pxsz', ... %%% img
    'bsz', 'bszum', 'bszcp', 'bszcpum'              ... %%% scalebar
    );
H=imsz(1);W=imsz(2);
Hcp=imszcp(1);Wcp=imszcp(2);


% (3) Para#2
% 1 cal

% 2 plot
%%% switch
swmaxfig = 1;
optight='loose';
ps_qs=[0.1, 0.8];% cell quest box location

%%% img
load('const_sh');
imgco_r=COR.tomato;
imgco_g=COG.lawngreen;
imgco_b=COB.skyblue;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [2] Process
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
        flebndmat = fleBNDmatc{i};
        flebndfig = fleBNDfigc{i};
        flebndfg1 = fleBNDfg1c{i};
        flebndfg2 = fleBNDfg2c{i};
        flebndfg3 = fleBNDfg3c{i};
        
        % (2) Load
        %%% img
        im_r=imreadfle(fleim_r,'tif');imaj_r=imad(im_r,sat, gam);imco_r=imgray2rgb(imaj_r,imgco_r);
        im_g=imreadfle(fleim_g,'tif');imaj_g=imad(im_g,sat, gam);imco_g=imgray2rgb(imaj_g,imgco_g);
        im_b=imreadfle(fleim_b,'tif');imaj_b=imad(im_b,sat, gam);imco_b=imgray2rgb(imaj_b,imgco_b);
        
        % (3) Cal
        if ctcal==0 || (ctcal==1 && ( strcmpi(opcal,'a') || strcmpi(opcal,'r') || strcmpi(opcal,'d')))
            [tbdc, rbdc, thsv]=loadfle(flebndmat,'mat', 'tbdc', 'rbdc', 'thsv');
        end
        if ctcal==1
            im=im_g;
            if strcmpi(opcal,'n')
                [tbdc, rbdc, thsv] =bdths_nc(im, 'swplotbdo', 0, 'swmagrec', 0, 'swrbdo', 0, 'swmaxfig', swmaxfig, ...
                    'plotbdoc', [], 'rbdoc', [], 'tl', ex, ...
                    'lnco', lnco, 'rlnco', lnco, 'lnwd', lnwd, 'faap', 0.2, 'imgco', imgco, 'gam', gam, 'txsz', txsz, 'sat', sat);
            end
            if strcmpi(opcal,'a')
                [tbdc, rbdc, thsv] =bdths_ac(im, tbdc, rbdc, thsv, [], [], [], 'swplotbdo', 0, 'swmagrec', 0, 'swrbdo', 0, 'swmaxfig', swmaxfig, ...
                    'plotbdoc', [], 'rbdoc', [], 'tl', ex, ...
                    'lnco', lnco, 'rlnco', lnco, 'lnwd', lnwd, 'faap', 0.2, 'imgco', imgco, 'gam', gam, 'txsz', txsz, 'sat', sat);
            end
            if strcmpi(opcal,'r')
                [tbdc, rbdc, thsv] =bdths_rc(im, tbdc, rbdc, thsv, [], [], [], 'swplotbdo', 1, 'swmagrec', 0, 'swrbdo', 0, 'swmaxfig', swmaxfig, ...
                    'plotbdoc', tbdc, 'rbdoc', rbdc, 'tl', ex, ...
                    'lnco', lnco, 'rlnco', lnco, 'lnwd', lnwd, 'faap', 0.2, 'imgco', imgco, 'gam', gam, 'txsz', txsz, 'sat', sat);
            end
            if strcmpi(opcal,'d')
                [tbdc, rbdc, thsv] =bdths_dc(im, tbdc, rbdc, thsv, [], [], [], 'swmaxfig', swmaxfig, ...
                    'tl', ex, ...
                    'lnco', lnco, 'rlnco', lnco, 'lnwd', lnwd, 'faap', 0.2, 'imgco', imgco, 'gam', gam, 'txsz', txsz, 'sat', sat);
            end
            savefle(flebndmat,'mat', tbdc, rbdc, thsv);
        end
        %%% bnd
        bdc=tbdc;
        [~, bwc]=bd2bwmaxc(bdc, imsz);
        nbd=length(bdc);
        
        % (4) Plot
        % 1 FA
        if ctplot==1
%             im=imco_r;
%             imshow(im);hold on;
%             if swbd==1;if ~isempty(bdc);plotbdc(bdc, 'lnty',lnty, 'lnwd', lnwd, 'lnco', [lnco, lnap], 'swtx', swtx, 'txsz', txsz, 'txco', [txco, txap]);end;end
%             if swscb==1;addscb([W*(1-0.03)-bsz, H*(1-0.05)], bsz, 'bco', [bco, bap], 'bwd', bwd);end
%             axcon1('xlm', [0 W], 'ylm', [0 H], 'xt', [], 'yt', [], 'optight', optight);
%             exportgraphics(gca,sprintf('%s.jpg',flebndfig),'Resolution',300);hold off;close;
            
            im=imco_g;
            imshow(im);hold on;
            if swbd==1;if ~isempty(bdc);plotbdc(bdc, 'lnty',lnty, 'lnwd', lnwd, 'lnco', [lnco, lnap], 'swtx', swtx, 'txsz', txsz, 'txco', [txco, txap]);end;end
            if swscb==1;addscb([W*(1-0.03)-bsz, H*(1-0.05)], bsz, 'bco', [bco, bap], 'bwd', bwd);end
            axcon1('xlm', [0 W], 'ylm', [0 H], 'xt', [], 'yt', [], 'optight', optight);
            exportgraphics(gca,sprintf('%s_g.jpg',flebndfig),'Resolution',300);hold off;close;
            
        end
        
        % 2 FA crop
        if ctplcp==1
%             % %%% im
%             im=imaj_r;
%             imgco=imgco_r;
%             for k=1:nbd
%                 % 1 para
%                 % %%% bd
%                 if ~isempty(bdc)
%                     bd = bdc{k};
%                     bw = bwc{k};
%                     % %%% im bd
%                     imbd = getimbw(im, bw);
%                     imco=imgray2rgb(imbd,imgco);
%                 end
%                 % %%% crop
%                 [xlm,ylm, ~, ~]=recbd(bd, imszcp, imsz, pxsz);
%                 
%                 % 2 plot
%                 imshow(imco, 'Border', 'tight');hold on;
%                 if ~isempty(bdc);if swbd==1;plotbdc(bd, 'lnty', lnty, 'lnwd', lnwd, 'lnco', [lnco, lnap]);end;end
%                 if swscb==1;addscb([xlm(2)-bszcp-Wcp*0.05, ylm(2)-Hcp*0.05], bszcp, 'bco', [bco, bap], 'bwd', bwd);end
%                 axcon1('xlm', xlm, 'ylm', ylm, 'xt', [], 'yt', [], 'optight', optight);
%                 exportgraphics(gca,sprintf('%s-%d.jpg',flebndfg1, k),'Resolution',300);hold off;close;
%             end

            % %%% im
            im=imaj_g;
            imgco=imgco_g;
            for k=1:nbd
                % 1 para
                % %%% bd
                if ~isempty(bdc)
                    bd = bdc{k};
                    bw = bwc{k};
                    % %%% im bd
                    imbd = getimbw(im, bw);
                    imco=imgray2rgb(imbd,imgco);
                end
                % %%% crop
                [xlm,ylm, ~, ~]=recbd(bd, imszcp, imsz, pxsz);
                
                % 2 plot
                imshow(imco, 'Border', 'tight');hold on;
                if ~isempty(bdc);if swbd==1;plotbdc(bd, 'lnty', lnty, 'lnwd', lnwd, 'lnco', [lnco, lnap]);end;end
                if swscb==1;addscb([xlm(2)-bszcp-Wcp*0.05, ylm(2)-Hcp*0.05], bszcp, 'bco', [bco, bap], 'bwd', bwd);end
                axcon1('xlm', xlm, 'ylm', ylm, 'xt', [], 'yt', [], 'optight', optight);
                exportgraphics(gca,sprintf('%s_g-%d.jpg',flebndfg1, k),'Resolution',300);hold off;close;
            end

        end
    end
end
end