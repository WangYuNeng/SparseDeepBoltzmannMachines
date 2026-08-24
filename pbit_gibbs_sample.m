function S = pbit_gibbs_sample(J_bipolar, h_bipolar, Groups, beta, num_sweeps, test_randoms)
%PBIT_GIBBS_SAMPLE Chromatic (graph-colored) block Gibbs sampling with
%simulated annealing, batched over independent chains.
%
%   S = PBIT_GIBBS_SAMPLE(J_bipolar, h_bipolar, Groups, beta, num_sweeps)
%   S = PBIT_GIBBS_SAMPLE(..., test_randoms)
%
%   J_bipolar : NM x NM sparse coupling matrix
%   h_bipolar : NM x B bias matrix (one column per independent chain, e.g.
%               one column per image being generated with its own label
%               clamp)
%   Groups    : 1 x required_colors cell array of p-bit index groups; bits
%               within a group are updated simultaneously, groups are
%               updated in sequence (this is what makes the graph
%               coloring correct: no two bits in the same group are
%               coupled, so their conditional distributions are
%               independent given everything else)
%   beta      : row vector, annealing (inverse temperature) schedule
%   num_sweeps: sweeps to run at each beta value
%   test_randoms : optional struct used ONLY to cross-check this sampler
%               against a ported implementation in another language.
%               Fields:
%                 .init  : NM x B matrix substituted for the initial
%                          2*rand(NM,B)-1 draw
%                 .draws : 1 x (length(beta)*num_sweeps*required_colors)
%                          cell array substituted, in order, for each
%                          2*rand(ng(c),B) draw inside the (beta, sweep,
%                          color) loop nest below
%               When omitted, behavior is unchanged (uses rand()).
%
%   Returns S, the NM x B final bipolar (+-1) state after the full
%   annealing schedule, one column per chain.

if nargin < 6
    test_randoms = [];
end
use_test_randoms = ~isempty(test_randoms);

NM = size(h_bipolar, 1);
B  = size(h_bipolar, 2);
required_colors = length(Groups);

% Precompute the per-color sparse sub-blocks and bias slices once, instead
% of re-slicing J_bipolar/h_bipolar on every one of the
% length(beta)*num_sweeps*required_colors iterations.
Jg = cell(1, required_colors);
hg = cell(1, required_colors);
ng = zeros(1, required_colors);
for c = 1:required_colors
    Jg{c} = J_bipolar(Groups{c}, :);
    hg{c} = h_bipolar(Groups{c}, :);
    ng(c) = length(Groups{c});
end

if use_test_randoms
    S = test_randoms.init;
else
    S = 2*rand(NM, B) - 1;
end

draw_idx = 0;
for kk = 1:length(beta)
    bkk = beta(kk);
    for k = 1:num_sweeps
        for c = 1:required_colors
            idx = Groups{c};
            x = bkk * (Jg{c}*S + hg{c});
            if use_test_randoms
                draw_idx = draw_idx + 1;
                r = test_randoms.draws{draw_idx};
            else
                r = rand(ng(c), B);
            end
            S(idx,:) = sign(tanh(x) - 2*r + 1);
        end
    end
end
end
