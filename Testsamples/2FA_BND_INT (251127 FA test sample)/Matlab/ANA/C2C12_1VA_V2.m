function C2C12_1VA_V1
% Target: to save variables, file names, and computational parameters
% Variable function - to save variables and file names
% By Sung Sik Hur (sstahur@gmail.com)
%   Last Modified: 220419

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [1] Pre-Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;close all;load('const_sh');
fprintf('########################\n### Saving Variables ###\n########################\n');
dirvar='1VA';

% (1) Control
exc={
'e1s1'	'P0_s1'	1	1
'e1s2'	'P0_s2'	1	2
'e2s1'	'P1_s1'	2	1
'e2s2'	'P1_s2'	2	2
};
nr=size(exc,1);

% (2) Image
fleimgc={
'..\EXP\TIF\e1'	'S01_3'	'S01_1'	'S01_0'
'..\EXP\TIF\e1'	'S02_3'	'S02_1'	'S02_0'
'..\EXP\TIF\e2'	'S01_3'	'S01_1'	'S01_0'
'..\EXP\TIF\e2'	'S02_3'	'S02_1'	'S02_0'
};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% [2] Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) File name
% 1 define
%%% img
fleIMGc                                                             = cell(nr, size(fleimgc,2)-1); % image Z Zref
%%% BND act
[fleBNDmatc, fleBNDfigc, fleBNDfg1c, fleBNDfg2c, fleBNDfg3c]        = deal(cell(nr, 1));
%%% BND nuc
[fleBD1matc, fleBD1figc, fleBD1fg1c, fleBD1fg2c, fleBD1fg3c]        = deal(cell(nr, 1));
%%% BND FAs
[fleBD2matc, fleBD2figc, fleBD2fg1c, fleBD2fg2c, fleBD2fg3c]        = deal(cell(nr, 1));
%%% MOR act
[fleMORmatc, fleMORfigc, fleMORfg1c, fleMORfg2c, fleMORfg3c]       = deal(cell(nr, 1));
%%% MOR nuc
[fleMO1matc, fleMO1figc, fleMO1fg1c, fleMO1fg2c, fleMO1fg3c]       = deal(cell(nr, 1));
%%% MOR FA
[fleMO2matc, fleMO2figc, fleMO2fg1c, fleMO2fg2c, fleMO2fg3c]       = deal(cell(nr, 1));
%%% INT act
[fleINTmatc, fleINTfigc, fleINTfg1c, fleINTfg2c, fleINTfg3c]       = deal(cell(nr, 1));
%%% INT nuc
[fleIT1matc, fleIT1figc, fleIT1fg1c, fleIT1fg2c, fleIT1fg3c]       = deal(cell(nr, 1));
%%% INT FA
[fleIT2matc, fleIT2figc, fleIT2fg1c, fleIT2fg2c, fleIT2fg3c]       = deal(cell(nr, 1));

% 2 initialize
for i=1:nr
    ex=exc{i,1};
    % 1 IMG
    fleIMGc{i,1}=fullfile(fleimgc{i,1},sprintf('%s%d',fleimgc{i,2}));
    fleIMGc{i,2}=fullfile(fleimgc{i,1},sprintf('%s%d',fleimgc{i,3}));
    fleIMGc{i,3}=fullfile(fleimgc{i,1},sprintf('%s%d',fleimgc{i,4}));
    
    % 2.1 BND - cell
    fleBNDmatc{i}=fullfile('BND Cell/mat', sprintf('BND_%s', ex)); % mat
    fleBNDfigc{i}=fullfile('BND Cell/fig', sprintf('BND_%s', ex)); % fig
    fleBNDfg1c{i}=fullfile('BND Cell/fig cp', sprintf('BND1_%s', ex)); % fig
    fleBNDfg2c{i}=fullfile('BND Cell/fig cp', sprintf('BND2_%s', ex)); % fig
    fleBNDfg3c{i}=fullfile('BND Cell/fig cp', sprintf('BND3_%s', ex)); % fig
    
    % 2.2 BND - nuc
    fleBD1matc{i}=fullfile('BND Nuc/mat', sprintf('BNDa_%s', ex)); % mat
    fleBD1figc{i}=fullfile('BND Nuc/fig', sprintf('BNDa_%s', ex)); % fig
    fleBD1fg1c{i}=fullfile('BND Nuc/fig cp', sprintf('BNDa1_%s', ex)); % fig
    fleBD1fg2c{i}=fullfile('BND Nuc/fig cp', sprintf('BNDa2_%s', ex)); % fig
    fleBD1fg3c{i}=fullfile('BND Nuc/fig cp', sprintf('BNDa3_%s', ex)); % fig
    
    % 2.3 BND - FAs
    fleBD2matc{i}=fullfile('BND FA/mat', sprintf('BNDb_%s', ex)); % mat
    fleBD2figc{i}=fullfile('BND FA/fig', sprintf('BNDb_%s', ex)); % fig
    fleBD2fg1c{i}=fullfile('BND FA/fig cp', sprintf('BNDb1_%s', ex)); % fig
    fleBD2fg2c{i}=fullfile('BND FA/fig cp', sprintf('BNDb2_%s', ex)); % fig
    fleBD2fg3c{i}=fullfile('BND FA/fig cp', sprintf('BNDb3_%s', ex)); % fig
    
    % 3.1 MOR - act
    fleMORmatc{i}=fullfile('MOR Cell/mat', sprintf('MOR_%s', ex)); % mat
    fleMORfigc{i}=fullfile('MOR Cell/fig', sprintf('MOR_%s', ex)); % fig
    fleMORfg1c{i}=fullfile('MOR Cell/fig cp', sprintf('MOR1_%s', ex)); % fig
    fleMORfg2c{i}=fullfile('MOR Cell/fig cp', sprintf('MOR2_%s', ex)); % fig
    fleMORfg3c{i}=fullfile('MOR Cell/fig cp', sprintf('MOR3_%s', ex)); % fig

    % 3.2 MO1 - nuc
    fleMO1matc{i}=fullfile('MOR Nuc/mat', sprintf('MO1_%s', ex)); % mat
    fleMO1figc{i}=fullfile('MOR Nuc/fig', sprintf('MO1_%s', ex)); % fig
    fleMO1fg1c{i}=fullfile('MOR Nuc/fig cp', sprintf('MO1a1_%s', ex)); % fig
    fleMO1fg2c{i}=fullfile('MOR Nuc/fig cp', sprintf('MO1a2_%s', ex)); % fig
    fleMO1fg3c{i}=fullfile('MOR Nuc/fig cp', sprintf('MO1a3_%s', ex)); % fig
    
    % 3.2 MO2 - FA
    fleMO2matc{i}=fullfile('MOR FA/mat', sprintf('MO2_%s', ex)); % mat
    fleMO2figc{i}=fullfile('MOR FA/fig', sprintf('MO2_%s', ex)); % fig
    fleMO2fg1c{i}=fullfile('MOR FA/fig cp', sprintf('MO2a1_%s', ex)); % fig
    fleMO2fg2c{i}=fullfile('MOR FA/fig cp', sprintf('MO2a2_%s', ex)); % fig
    fleMO2fg3c{i}=fullfile('MOR FA/fig cp', sprintf('MO2a3_%s', ex)); % fig
    
    % 4.1 INT
    fleINTmatc{i}=fullfile('INT Cell/mat', sprintf('INT_%s', ex)); % mat
    fleINTfigc{i}=fullfile('INT Cell/fig', sprintf('INT_%s', ex)); % fig
    fleINTfg1c{i}=fullfile('INT Cell/fig cp', sprintf('INT1_%s', ex)); % fig
    fleINTfg2c{i}=fullfile('INT Cell/fig cp', sprintf('INT2_%s', ex)); % fig
    fleINTfg3c{i}=fullfile('INT Cell/fig cp', sprintf('INT3_%s', ex)); % fig
    
    % 4.2 IT1
    fleIT1matc{i}=fullfile('INT Nuc/mat', sprintf('IT1_%s', ex)); % mat
    fleIT1figc{i}=fullfile('INT Nuc/fig', sprintf('IT1_%s', ex)); % fig
    fleIT1fg1c{i}=fullfile('INT Nuc/fig cp', sprintf('IT1a1_%s', ex)); % fig
    fleIT1fg2c{i}=fullfile('INT Nuc/fig cp', sprintf('IT1a2_%s', ex)); % fig
    fleIT1fg3c{i}=fullfile('INT Nuc/fig cp', sprintf('IT1a3_%s', ex)); % fig
    
    % 4.2 IT2
    fleIT2matc{i}=fullfile('INT FA/mat', sprintf('IT2_%s', ex)); % mat
    fleIT2figc{i}=fullfile('INT FA/fig', sprintf('IT2_%s', ex)); % fig
    fleIT2fg1c{i}=fullfile('INT FA/fig cp', sprintf('IT2a1_%s', ex)); % fig
    fleIT2fg2c{i}=fullfile('INT FA/fig cp', sprintf('IT2a2_%s', ex)); % fig
    fleIT2fg3c{i}=fullfile('INT FA/fig cp', sprintf('IT2a3_%s', ex)); % fig
end
fleMORxls=fullfile('MOR Cell/xls', 'MOR'); % xls
fleMO1xls=fullfile('MOR Nuc/xls',  'MO1'); % xls
fleMO2xls=fullfile('MOR FA/xls',   'MO2'); % xls
fleINTxls=fullfile('INT Cell/xls', 'INT'); % xls
fleIT1xls=fullfile('INT Nuc/xls',  'IT1'); % xls
fleIT2xls=fullfile('INT FA/xls',   'IT2'); % xls

%%%% [2] experimental constants
% 1 load

% 2 cal
%%% img
pxsz=[0.112 0.112, 1.0]; 
imsz=[2200 3200];
imszum=imsz.*pxsz(1:2);
imszcpum=[150 150]; % crop image size in um
imszcp=imszcpum./pxsz(1:2);

%%% FA size
arum_sm= 0.33; % minimum area in um^2  ★
arum_lg= 30; % maximum area in um^2  ★
ar_sm= round(arum_sm /pxsz(1)/pxsz(2)); % minimum area in px
ar_lg= round(arum_lg /pxsz(1)/pxsz(2)); % maximum area in px

% 3 plot
%%% scale bar 
bszum=100;     % scalebar size - um
bszcpum=50;    % scalebar size for crop - um
bsz=bszum/pxsz(1);
bszcp=bszcpum/pxsz(1);

% (3) Save
% 1 control
savefle(fullfile(dirvar,'CTL'),'mat', exc); %%% exc

% 2 file
savefle(fullfile(dirvar,'FLE'),'mat', ...
fleIMGc, ...
fleBNDmatc, fleBNDfigc, fleBNDfg1c, fleBNDfg2c, fleBNDfg3c,                         ... %%% bnd
fleBD1matc, fleBD1figc, fleBD1fg1c, fleBD1fg2c, fleBD1fg3c,                         ... %%% bnd1
fleBD2matc, fleBD2figc, fleBD2fg1c, fleBD2fg2c, fleBD2fg3c,                         ... %%% bnd2
fleMORmatc, fleMORfigc, fleMORfg1c, fleMORfg2c, fleMORfg3c, fleMORxls,              ... %%% mor
fleMO1matc, fleMO1figc, fleMO1fg1c, fleMO1fg2c, fleMO1fg3c, fleMO1xls,              ... %%% mor1
fleMO2matc, fleMO2figc, fleMO2fg1c, fleMO2fg2c, fleMO2fg3c, fleMO2xls,              ... %%% mor1
fleINTmatc, fleINTfigc, fleINTfg1c, fleINTfg2c, fleINTfg3c, fleINTxls,              ... %%% int
fleIT1matc, fleIT1figc, fleIT1fg1c, fleIT1fg2c, fleIT1fg3c, fleIT1xls,              ... %%% int1
fleIT2matc, fleIT2figc, fleIT2fg1c, fleIT2fg2c, fleIT2fg3c, fleIT2xls               ... %%% int2
);

% 3 data
savefle(fullfile(dirvar,'DAT'),'mat',   ...                 
imsz, imszum, imszcp, imszcpum, pxsz, ... %%% img
bsz, bszum, bszcp, bszcpum,           ... %%% scalebar
arum_sm, arum_lg, ar_sm, ar_lg        ... %%% FA size filter
);
end

