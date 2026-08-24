function test_pbit_equivalence(label, batch_size, num_sweeps, test_randoms_path, output_path)
%TEST_PBIT_EQUIVALENCE Cross-language equivalence harness for
%pbit_gibbs_sample.m, invoked by python/test_equivalence.py.
%
%   Builds J_bipolar/H/Groups exactly like generate_labeled_batch.m, but
%   substitutes an externally generated random sequence (see the
%   test_randoms hook in pbit_gibbs_sample.m) so the run is deterministic
%   and directly comparable to the Python/JAX port, independent of the two
%   languages' different (non-interchangeable) PRNGs.

beta = 0:0.125:5; % annealing schedule (same as generate_labeled_batch.m)

colormap1 = readmatrix('colorMap_4264.csv');
required_colors = length(unique(colormap1));
Groups = cell(1,required_colors);
for k = 1:required_colors
    Groups{k} = find(colormap1==k);
end

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
NM = size(J_bipolar, 1);

hclamp = zeros(1,NM);
hclamp(index_sticker) = -1000;
hclamp(index_sticker(:,label+1)) = 1000;
h_col = (hout + hclamp)';
H = repmat(h_col, 1, batch_size);

test_randoms = load(test_randoms_path);

S = pbit_gibbs_sample(J_bipolar, H, Groups, beta, num_sweeps, test_randoms);

img = reshape(S(index_visible, 1)', [28,28]);

save(output_path, 'S', 'img');
end
