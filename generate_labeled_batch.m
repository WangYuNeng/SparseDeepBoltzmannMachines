function generate_labeled_batch(label, seed, batch_size, start_idx, output_dir, num_sweeps)
%GENERATE_LABELED_BATCH Generate a batch of p-bit DBM samples conditioned
%only on a digit label (0-9) -- no specific MNIST example is used, since
%the sampler never clamps to the visible pixels, only to the label
%("sticker") bits (see Image_generation.m).
%
%   generate_labeled_batch(label, seed, batch_size, start_idx, output_dir)
%   generate_labeled_batch(..., num_sweeps)  % optional, for smoke-testing
%
%   label      : digit class to condition on (0-9)
%   seed       : RNG seed for this task. MUST be unique per task: MATLAB's
%                global random stream is NOT auto-randomized at startup,
%                so every fresh `matlab -batch` process would otherwise
%                draw the identical "random" sequence and every task
%                would generate identical images for a given label.
%   batch_size : number of images to generate in this call
%   start_idx  : 0-based index of the first image in this batch, used only
%                to name output files so chunks from different tasks
%                don't collide
%   output_dir : base output directory; images are written to
%                <output_dir>/label_<label>/sample_<00000+start_idx>.png
%   num_sweeps : optional override of the annealing sweep count per beta
%                value (default 10000, matching Image_generation.m).
%                Only meant for quick smoke tests of this script.

if nargin < 6 || isempty(num_sweeps)
    num_sweeps = 10000;
end

rng(seed, 'twister');

beta = 0:0.125:5; % annealing schedule (same as Image_generation.m)

load('JJ_4264.mat')
colormap1 = readmatrix('colorMap_4264.csv');
required_colors = length(unique(colormap1));
Groups = cell(1,required_colors);
for k = 1:required_colors
    Groups{k} = find(colormap1==k);
end
NM = length(W);

load Jout_100.mat
load hout_100.mat
load index_visible.mat
load index_sticker1.mat
load index_sticker2.mat
load index_sticker3.mat
load index_sticker4.mat
load index_sticker5.mat
index_sticker = [index_sticker1; index_sticker2; index_sticker3; index_sticker4; index_sticker5];

J_bipolar = sparse(Jout);

hclamp = zeros(1,NM);
hclamp(index_sticker) = -1000;
hclamp(index_sticker(:,label+1)) = 1000;
h_col = (hout + hclamp)';
H = repmat(h_col, 1, batch_size);

fprintf('[label=%d seed=%d] generating %d images (start_idx=%d, num_sweeps=%d)...\n', ...
    label, seed, batch_size, start_idx, num_sweeps);
tic
S = pbit_gibbs_sample(J_bipolar, H, Groups, beta, num_sweeps);
toc

out_dir = fullfile(output_dir, sprintf('label_%d', label));
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

for b = 1:batch_size
    img = reshape(S(index_visible, b)', [28,28]);
    img01 = (img + 1) / 2; % bipolar {-1,+1} -> {0,1} for imwrite
    fname = fullfile(out_dir, sprintf('sample_%05d.png', start_idx + b - 1));
    imwrite(img01, fname);
end

fprintf('[label=%d] wrote %d images to %s\n', label, batch_size, out_dir);
end
