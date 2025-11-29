%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C2C12_mod_MOR_FA_F3_V1(ctob, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Module to determine morphology of FAs
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
%   Last Modified - 220420

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% [1] Pre-Process
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Default
% 1 default
co_gr=0.5*[1 1 1];
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

% (2) Control
ctall=logical(sum([ctcal, ctplot, ctplcp, ctxls]));if ctall==0; return; end %%% check control

% (3) Load
% 1 control
exc=loadfle(flectl,'mat', 'exc');

% 2 file
[fleIMGc,                                                                       ... %%% img
fleBNDmatc, fleBNDfigc, fleBNDfg1c, fleBNDfg2c, fleBNDfg3c,                     ... %%% bnd
fleBD2matc, fleBD2figc, fleBD2fg1c, fleBD2fg2c, fleBD2fg3c,                     ... %%% bnd1
fleMORmatc, fleMORfigc, fleMORfg1c, fleMORfg2c, fleMORfg3c, fleMORxls,          ... %%% mor
fleMO2matc, fleMO2figc, fleMO2fg1c, fleMO2fg2c, fleMO2fg3c, fleMO2xls           ... %%% mor1
]...
=loadfle(flefle,'mat', ...
'fleIMGc',                                                                      ... %%% img
'fleBNDmatc', 'fleBNDfigc', 'fleBNDfg1c', 'fleBNDfg2c', 'fleBNDfg3c',           ... %%% bnd
'fleBD2matc', 'fleBD2figc', 'fleBD2fg1c', 'fleBD2fg2c', 'fleBD2fg3c',           ... %%% bnd1
'fleMORmatc', 'fleMORfigc', 'fleMORfg1c', 'fleMORfg2c', 'fleMORfg3c','fleMORxls',... %%% mor
'fleMO2matc', 'fleMO2figc', 'fleMO2fg1c', 'fleMO2fg2c', 'fleMO2fg3c', 'fleMO2xls'... %%% mor1
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
swmaxfig = 0;
optight='loose';
swell=1;

%%% img
load('const_sh');
imgco_r = COR.tomato;
imgco_g = COG.lightgreen;
imgco_b = COB.skyblue;
satd=[0.001 0.999];
faap=0.2;

%%% mor
molnco=COB.royalblue;
molnty='-';
molnwd=2;
momksz=5;
mofaco=COB.royalblue;
ellfaco=imgco_r;
ellfaap=0.2;
ellegco='none';
ellegap=0.1;
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
        flebndmat = fleBNDmatc{i}; % Cell
        flebd2mat = fleBD2matc{i}; % FA
        %%% mor
        flemormat = fleMORmatc{i}; % Cell
        flemo2mat = fleMO2matc{i}; % FA
        flemo2fig = fleMO2figc{i};
        flemo2fg1 = fleMO2fg1c{i};
        
        % (2) Load
        %%% img
        im_r=imreadfle(fleim_r,'tif');imaj_r=imad(im_r,sat, gam);imco_r=imgray2rgb(imaj_r,imgco_r);
        im_g=imreadfle(fleim_g,'tif');imaj_g=imad(im_g,sat, gam);imco_g=imgray2rgb(imaj_g,imgco_g);
        im_b=imreadfle(fleim_b,'tif');imaj_b=imad(im_b,sat, gam);imco_b=imgray2rgb(imaj_b,imgco_b);
        
        %%% bnd - cell
        [tbdc, rbdc, thsv]=loadfle(flebndmat,'mat', 'tbdc', 'rbdc', 'thsv'); % Cell
        %%% bnd - FA
        [bdokcc, bwokc, stokc,              ... %%% FA ok
                rbdc, thsv,                 ... %%% rbd & ths
                bdlgcc, bwlgc, stlgc,       ... %%% FA large
                bdsmcc, bwsmc, stsmc        ... %%% FA small
            ]=loadfle(flebd2mat,'mat',      ...
                'bdokcc', 'bwokc', 'stokc', ... %%% FA ok
                'rbdc', 'thsv',               ... %%% rbd & ths
                'bdlgcc', 'bwlgc', 'stlgc',  ... %%% FA large
                'bdsmcc', 'bwsmc', 'stsmc'   ... %%% FA small
                );
        bdc_cell = tbdc;
        [~, bwc_cell]=bd2bwmaxc(tbdc, imsz);
        bdcc_FA = bdokcc;
        bwc_FA  = bwokc;
        nbd=length(tbdc);

        % (3) Cal
        if ctcal==0
            [ctc,  arc,  lnac,  lnbc,  elc,  crc,  rfc, orc,   ctumc,  arumc,  lnaumc,  lnbumc, ...
             armv, lnamv, lnbmv, elmv, crmv, rfmv, ormv,       arummv, lnaummv, lnbummv, ...
             arsv, arumsv] = loadfle(flemo2mat,'mat', ...
            'ctc','arc','lnac','lnbc','elc','crc', 'rfc', 'orc',  'ctumc','arumc','lnaumc','lnbumc', ...
            'armv','lnamv','lnbmv','elmv','crmv', 'rfmv', 'ormv', 'arummv','lnaummv','lnbummv', ...
            'arsv', 'arumsv');
        end
        if ctcal==1
            [ctc, arc, lnac, lnbc, elc, crc, rfc, orc, ...   
             ctumc, arumc, lnaumc, lnbumc, armv,lnamv,lnbmv,elmv,crmv,rfmv, ormv,  ...      
             arummv,lnaummv,lnbummv, arsv, arumsv] = calmofabwc(bwokc, pxsz);
            savefle(flemo2mat,'mat', ctc, arc, lnac, lnbc, elc, crc, rfc, orc, ctumc, arumc, lnaumc, lnbumc, ...
             armv, lnamv, lnbmv, elmv, crmv, rfmv, ormv,        ...
             arummv, lnaummv, lnbummv, arsv, arumsv);
        end
        % (4) Plot
        if ctplot==1
            %%% img
            im=imaj_g;
            imshow(im, 'Border', 'tight');hold on;
            %%% bd
            bd1c  = bdc_cell;
            bw1c  = bwc_cell;
            bd2cc = bdcc_FA;
            bw2c  = bwc_FA;
            
            for k=1:nbd
                %%% bd
                bd1=bd1c{k};
                bd2c=bd2cc{k};
                %%% mo
                ct=ctc{k};
                ar=arumc{k};
                ars=sum(ar(:),'omitnan');
                if ~isempty(ct)
                    lna=lnac{k};
                    lnb=lnbc{k};
                    or = orc{k};
                    [xe, ye] =calellrot(ct(:,1), ct(:,2), lna, lnb, -or,'np', 360);
                    patch('XData', xe', 'YData', ye', 'FaceColor', ellfaco, 'FaceAlpha', ellfaap, 'LineStyle', '-', 'EdgeColor', ellegco, 'EdgeAlpha', ellegap); hold on;
                end
                if swbd==1;plotbdc(bd1,  'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd);end
                if swbd==1;plotbdc(bd2c, 'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd);end
                if swval==1;textxy(mean(bd1,1), sprintf('ar\n%0.0f',ars), 'txco', txco, 'txsz', txsz, 'txal', txal);end
            end
            if swscb;addscb([W*(1-0.03)-bsz, H*(1-0.05)], bsz, 'bco', [bco, bap], 'bwd', bwd);end
            exportgraphics(gca,sprintf('%s.jpg',flemo2fig),'Resolution',300);hold off;close;
        end
        % (5) Plot Crop
        if ctplcp==1
            %%% img
            im=imaj_g;
            %%% bd
            bd1c  = bdc_cell;
            bw1c  = bwc_cell;
            bd2cc = bdcc_FA;
            bw2c  = bwc_FA;
               
            for k=1:nbd
                %%% bd
                bd1  = bd1c{k};
                bw1  = bw1c{k}; 
                bd2c = bd2cc{k};
                bw2  = bw2c{k};
                %%% rec
                [xlm,ylm, xlmum, ylmum]=recbd(bd1, imszcp, imsz, pxsz);
                
                %%% mo
                imbd = getimbw(im, bw2);
                ct=ctc{k};
                ar=arumc{k};
                ars=sum(ar(:),'omitnan');
                
                %%% plot
                imshow(imbd, 'Border', 'tight');hold on;
                
                if swell==1
                if ~isempty(ct)
                    lna=lnac{k};
                    lnb=lnbc{k};
                    or = orc{k};
                    [xe, ye] =calellrot(ct(:,1), ct(:,2), lna, lnb, -or,'np', 360);
                    patch('XData', xe', 'YData', ye', 'FaceColor', ellfaco, 'FaceAlpha', ellfaap, 'LineStyle', '-', 'EdgeColor', ellegco, 'EdgeAlpha', ellegap); hold on;
                end
                end
                if swval==1;textxy(mean(bd1,1), sprintf('ar#%d\n%0.0f \mm^2',k, ars), 'txco', txco, 'txsz', txsz, 'txal', txal);end
                plotbdc(bd1, 'lnty', lnty, 'lnco', [lnco, lnap], 'lnwd', lnwd);hold on;
                if swscb==1;addscb([xlm(2)-bszcp-Wcp*0.05, ylm(2)-Hcp*0.05], bszcp, 'bco', [bco bap], 'bwd', bwd);end
                axcon1('xlm', xlm, 'ylm', ylm, 'xt', [], 'yt', [], 'optight', optight);
                exportgraphics(gca,sprintf('%s-%d.jpg',flemo2fg1, k),'Resolution',300);hold off;close;
            end
        end
        
        if ctxls==1
            for k=1:nbd
                ctum=ctumc{k};
                arum=arumc{k};
                lnum=lnaumc{k};
                el=elc{k};
                cr=crc{k};
                rf=rfc{k};
                or=orc{k};
                
                nob=length(arum);
                for m=1:nob
                    % object out
                    ii=ii+1;
                    lex{ii,1} = sprintf('e%ds%dc%do%d', exc{i,3}, exc{i,4}, k, m);
                    lep(ii,1) = exc{i,3};
                    lsp(ii,1) = exc{i,4};
                    lcp(ii,1) = k;
                    lop(ii,1) = m;
                    lcx(ii,1) = ctum(m,1);
                    lcy(ii,1) = ctum(m,2);
                    lar(ii,1) = arum(m);
                    lln(ii,1) = lnum(m);
                    lel(ii,1) = el(m);
                    lcr(ii,1) = cr(m);
                    lrf(ii,1) = rf(m);
                    lor(ii,1) = or(m);
                end
                % cell out
                jj=jj+1;
                vex{jj,1} = sprintf('e%ds%dc%d', exc{i,3}, exc{i,4}, k);
                vep(jj,1) = exc{i,3};
                vsp(jj,1) = exc{i,4};
                vcp(jj,1) = k;
                var(jj,1) = arummv(k);
                vln(jj,1) = lnaummv(k);
                vel(jj,1) = elmv(k);
                vcr(jj,1) = crmv(k);
                vrf(jj,1) = rfmv(k);
                if isnan(arummv(k))
                    vno(jj,1) = 0;
                else
                    vno(jj,1) = nob;
                end
                vars(jj,1) = arumsv(k);
            end
        end
    end
end
if ctxls==1
    flexls=fleMO2xls;
    % (1) LIST FA obj
    clear H1 H2 H3 B1 B2 B3 res;
    H1={'ex'};
    H2={'ep', 'sp','cp', 'op'};
    H3={'cx', 'cy', 'ar', 'ln', 'el', 'cr', 'rf', 'or'};
    B1=lex;
    B2=[lep, lsp, lcp, lop];B2=num2cell(B2);
    B3=[lcx, lcy, lar, lln, lel, lcr, lrf, lor];B3=num2cell(B3);
    res=[H1, H2, H3;B1, B2, B3];
    sht='ALL obj';saveflexls(flexls, 'xlsx', sht, res);
    
    % (2) LIST FA mean
    clear H1 H2 H3 B1 B2 B3 res;
    H1={'ex'};
    H2={'ep', 'sp','cp'};
    H3={'ar', 'ln', 'el', 'cr', 'rf', 'no', 'ars'};
    B1=vex;
    B2=[vep, vsp, vcp];B2=num2cell(B2);
    B3=[var, vln, vel, vcr, vrf, vno, vars];B3=num2cell(B3);
    res=[H1, H2, H3;B1, B2, B3];
    sht='ALL mean';saveflexls(flexls, 'xlsx', sht, res);
    
    % (3) TABLE FA obj
    clear H1 H2 H3 B1 B2 B3 res;
    % 1 data
    gv=[lep, lsp, lcp];
    g1=1:max(lep(:));n1=length(g1);
    g2=1:max(lsp(:));n2=length(g2);
    g3=1:max(lcp(:));n3=length(g3);
    gic={g1, g2, g3};
    [mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor, nm]=v2m_3Dmt(gv, gic, lep, lsp, lcp, lop, lcx, lcy, lar, lln, lel, lcr, lrf, lor);
    nmax=max(nm(:));
    [mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor]=permutemt([3 2 1 4],        mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor); %to change dimensions
    [mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor]=reshapemt([n1*n2*n3, nmax], mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor);
    [mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor]=transposemt(                mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor);
    
    % find nan col and remove
    iempty=logical(prod(isnan(mar),1)); % empty col index
    [mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor]=rmcolmt(iempty,               mep, msp, mcp, mop, mcx, mcy, mar, mln, mel, mcr, mrf, mor);
    
    % 2 head
    nr=size(mep,1);
    nc=size(mep,2);
    % head
    [H1, H3]=deal(cell(1,nc));
    H2={'n'};
    for j=1:nc
            H1{1,j}=sprintf('e%ds%dc%d', mep(1,j), msp(1,j), mcp(1,j));
            H3{1,j}=sprintf('e%ds%dc%d', mep(1,j), msp(1,j), mcp(1,j));
    end
    % 3. Bdy
    B1=cell(nr,nc);
    B2=cell(nr,1);
    for i=1:nr
        for j=1:nc
            if ~isnan(mar(1,j))
                B1{i,j}=sprintf('e%ds%dc%do%d',mep(i,j),msp(i,j), mcp(i,j), mop(i,j));
            end
        end
    end
    for i=1:nr
        B2{i,1}=sprintf('n%d', i);
    end
    % 4. write
    B3=num2cell(mep);sht='ep_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(msp);sht='sp_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mcp);sht='cp_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mop);sht='op_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mcx);sht='cx_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mcy);sht='cy_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mar);sht='ar_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mln);sht='ln_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mel);sht='el_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mcr);sht='cr_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mrf);sht='rf_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(mor);sht='or_o';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    
    % (4) TABLE FA mean
    clear H1 H2 H3 B1 B2 B3 res;
    % 1 data
    gv=vep;
    g1=1:max(vep(:));n1=length(g1);
    gi=g1;
    [pep, psp, pcp, par, pln, pel, pcr, prf, pno, pars]=v2m_1Dmt(gv, gi, vep, vsp, vcp, var, vln, vel, vcr, vrf, vno, vars);
    [pep, psp, pcp, par, pln, pel, pcr, prf, pno, pars]=transposemt(     pep, psp, pcp, par, pln, pel, pcr, prf, pno, pars);
    
    % find nan col and remove
    iempty=logical(prod(isnan(par),1)); % empty col index
    [pep, psp, pcp, par, pln, pel, pcr, prf, pno, pars]=rmcolmt(iempty,  pep, psp, pcp, par, pln, pel, pcr, prf, pno, pars);
    
    % calculate the number
    pnm=sum(~isnan(par),1); % mean
    
    % 2 head
    nr=size(pep,1);
    nc=size(pep,2);
    [H1, H3]=deal(cell(1,nc));
    H2={'n'};
    for j=1:nc
            H1{1,j}=sprintf('e%d', pep(1,j));
            H3{1,j}=sprintf('e%d', pep(1,j));
    end
    
    % 3. Body
    B1=cell(nr,nc);
    B2=cell(nr,1);
    for i=1:nr
        for j=1:nc
                B1{i,j}=sprintf('e%ds%dc%d',pep(i,j),psp(i,j), pcp(i,j));
        end
    end
    for i=1:nr
        B2{i,1}=sprintf('n%d', i);
    end
    % 4. write
    B3=num2cell(pep);sht='ep_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(psp);sht='sp_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(pcp);sht='cp_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(par);sht='ar_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(pln);sht='ln_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(pel);sht='el_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(pcr);sht='cr_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(prf);sht='rf_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(pno);sht='no_m';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(pars);sht='ar_s';  res=[H1,H2 H3;B1,B2,B3];saveflexls(flexls, 'xlsx', sht, res);
    B3=num2cell(pnm);sht='nm';  res=[H1,H2 H3;B1(1,:),B2(1,:),B3];saveflexls(flexls, 'xlsx', sht, res);
    
end
end
