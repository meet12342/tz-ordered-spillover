#!/usr/bin/env python3
"""Result figures, regenerated from the project's recorded numbers."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Circle

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.axisbelow': True})

# validated palette (dataviz default theme, light surface)
S1, S2, S3, S4, S5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8a86'
SURF = '#fcfcfb'
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
             '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('seqblue', BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list('div', ['#b8470f', '#eb6834', '#f5c4ad',
                                                '#f0efec', '#9ec5f4', '#2a78d6', '#184f95'])


def finish(fig, ax_list, fname, legend=None):
    for ax in (ax_list if isinstance(ax_list, (list, np.ndarray)) else [ax_list]):
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['left', 'bottom']:
            ax.spines[sp].set_color('#d7d6d1')
        ax.tick_params(colors=INK2, labelsize=8, length=3, color='#d7d6d1')
        ax.set_facecolor(SURF)
    fig.patch.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=220, facecolor=SURF,
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('wrote', fname)


MKT = ['TWSE', 'NKY', 'KS11', 'HSI', 'NIFIT', 'DAX', 'SX8P', 'NDX', 'REMX', 'SOX', 'GLD', 'UST']
RANK = {'TWSE': 1, 'NKY': 2, 'KS11': 3, 'HSI': 4, 'NIFIT': 5, 'DAX': 6, 'SX8P': 6,
        'NDX': 7, 'REMX': 7, 'SOX': 7}

# =================================================== F1  mask staircase
K = len(MKT)
M = np.zeros((K, K))
for a, mi in enumerate(MKT):
    for b, mj in enumerate(MKT):
        if mi in RANK and mj in RANK and RANK[mj] < RANK[mi]:
            M[a, b] = 1
fig, ax = plt.subplots(figsize=(5.3, 4.6))
ax.imshow(M, cmap=LinearSegmentedColormap.from_list('m', ['#f0efec', '#2a78d6']),
          vmin=0, vmax=1)
ax.set_xticks(range(K)); ax.set_xticklabels(MKT, rotation=90, fontsize=7.5)
ax.set_yticks(range(K)); ax.set_yticklabels(MKT, fontsize=7.5)
ax.set_xlabel('transmitter  $j$', fontsize=8.5, color=INK2)
ax.set_ylabel('receiver  $i$', fontsize=8.5, color=INK2)
ax.set_title(f'Admitted same-day edges: {int(M.sum())} of {K*(K-1)}',
             fontsize=9.5, color=INK, pad=8)
for s in range(K + 1):
    ax.axhline(s - .5, color='white', lw=.8); ax.axvline(s - .5, color='white', lw=.8)
ax.text(0.5, -0.30, 'Filled cells are permitted; the staircase is read off published closing times.',
        transform=ax.transAxes, ha='center', fontsize=7.2, color=INK2)
finish(fig, ax, 'f_mask.png')

# =================================================== F2  model accuracy
models = ['Random walk', 'AR(1)', 'HAR', 'Time-zone-ordered\ngraph', 'Unconstrained\ngraph']
rmse = [1.0687, 0.9253, 0.8610, 0.8152, 0.7593]
focal = [False, False, False, True, False]
fig, ax = plt.subplots(figsize=(6.6, 3.1))
cols = [S1 if f else '#9ec5f4' for f in focal]
b = ax.barh(range(len(models)), rmse, color=cols, height=.62, zorder=3)
for i, v in enumerate(rmse):
    ax.text(v + .012, i, f'{v:.4f}', va='center', fontsize=8,
            color=INK if focal[i] else INK2, weight='bold' if focal[i] else 'normal')
ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel('One-step RMSE  (lower is better)', fontsize=8.5, color=INK2)
ax.set_xlim(0, 1.21)
ax.grid(axis='x', ls=':', color='#e3e2dd', lw=.9)
ax.set_title('Walk-forward accuracy across four folds', fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_accuracy.png')

# =================================================== F3  fold stability
folds = [1, 2, 3, 4]
series = [('Random walk', [1.0581, 1.0515, 1.0494, 1.1157], S5),
          ('AR(1)', [0.8774, 0.9312, 0.8553, 1.0372], S4),
          ('HAR', [0.8771, 0.8428, 0.8145, 0.9096], S3),
          ('Ordered graph', [0.7989, 0.7956, 0.7895, 0.8767], S1),
          ('Unconstrained', [0.7698, 0.7239, 0.7350, 0.8084], S2)]
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for name, vals, c in series:
    ax.plot(folds, vals, '-o', color=c, lw=2, ms=6, mec='white', mew=1.2, label=name, zorder=3)
    ax.text(4.07, vals[-1], name, fontsize=7.6, color=c, va='center', weight='bold')
ax.set_xticks(folds); ax.set_xlim(.85, 5.25)
ax.set_xlabel('Walk-forward fold', fontsize=8.5, color=INK2)
ax.set_ylabel('RMSE', fontsize=8.5, color=INK2)
ax.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
ax.set_title('The ranking is identical in every fold', fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_folds.png')

# =================================================== F4  per-market vs HAR
vs = [2.2, 8.2, 11.1, -3.0, 5.9, 3.9, 7.5, 11.0, 6.7, 6.7, 2.4, 3.6]
order = np.argsort(vs)
fig, ax = plt.subplots(figsize=(6.6, 3.6))
cols = [S2 if vs[i] < 0 else S1 for i in order]
ax.barh(range(K), [vs[i] for i in order], color=cols, height=.62, zorder=3)
for p, i in enumerate(order):
    off = .35 if vs[i] >= 0 else -.35
    ax.text(vs[i] + off, p, f'{vs[i]:+.1f}%', va='center',
            ha='left' if vs[i] >= 0 else 'right', fontsize=7.8, color=INK2)
ax.set_yticks(range(K)); ax.set_yticklabels([MKT[i] for i in order], fontsize=8)
ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel('Improvement of the ordered model over HAR (%)', fontsize=8.5, color=INK2)
ax.set_xlim(-5.5, 14)
ax.grid(axis='x', ls=':', color='#e3e2dd', lw=.9)
ax.set_title('Eleven of twelve markets improve on HAR', fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_permarket.png')

# =================================================== F5  value of forbidden edges by rank
unmask = [5.7, 4.5, 3.9, 5.8, 0.1, 12.4, 11.5, 20.3, 3.9, 17.2, 4.5, 2.2]
rk = [RANK.get(m, 8) for m in MKT]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for r, u, m in zip(rk, unmask, MKT):
    c = S3 if r == 8 else S1
    ax.scatter(r, u, s=70, color=c, ec='white', lw=1.2, zorder=3)
    ax.text(r + .09, u, m, fontsize=7.2, va='center', color=INK2)
ax.set_xticks(range(1, 9))
ax.set_xticklabels(['1\n05:30', '2\n06:00', '3\n06:30', '4\n08:00', '5\n10:00',
                    '6\n16:30', '7\n21:00', 'cont.'], fontsize=7.5)
ax.set_xlabel('Closing rank (UTC close)', fontsize=8.5, color=INK2)
ax.set_ylabel('Gain from unmasking (%)', fontsize=8.5, color=INK2)
ax.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
ax.set_xlim(.5, 8.6)
ax.set_title('The value of forbidden information rises with closing rank',
             fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_rankgain.png')

# =================================================== F6  nested ablation
labels = ['No graph', 'Shuffled\ncontrol', 'Strict clock\norder',
          '+ simultaneous,\ncontinuous', '+ backward\n(unconstrained)']
gain = [0.0, 3.21, 5.60, 8.92, 11.49]
edges = [0, 132, 41, 71, 132]
cols = ['#cfcec9', '#9ec5f4', S1, S3, S2]
fig, ax = plt.subplots(figsize=(6.8, 3.5))
ax.bar(range(5), gain, color=cols, width=.6, zorder=3)
for i, (g, e) in enumerate(zip(gain, edges)):
    if i:
        ax.text(i, g + .22, f'+{g:.2f}', ha='center', fontsize=8.2, color=INK, weight='bold')
    ax.text(i, -.75, f'{e} edges', ha='center', fontsize=7.2, color=MUTED)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=7.8)
ax.set_ylabel('Gain over a graph-free model\n(percentage points)', fontsize=8.5, color=INK2)
ax.set_ylim(-1.3, 13)
ax.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
ax.annotate('', xy=(4, 12.2), xytext=(2, 12.2),
            arrowprops=dict(arrowstyle='<->', color=INK2, lw=1.1))
ax.text(3, 12.5, 'binary contrast attributes all of this to look-ahead',
        ha='center', fontsize=7.2, color=INK2)
ax.set_title('Only the final step admits genuinely unavailable information',
             fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_ablation.png')

# =================================================== F7  attribution split
parts = [('Capacity\n(shuffled control)', 3.21, '#cfcec9'),
         ('Admissible edges', 2.39, S1),
         ('Simultaneous &\ncontinuous', 3.32, S3),
         ('Genuine look-ahead', 2.57, S2)]
fig, ax = plt.subplots(figsize=(6.8, 2.1))
left = 0
for name, v, c in parts:
    ax.barh(0, v, left=left, color=c, height=.5, zorder=3, ec=SURF, lw=2)
    ax.text(left + v / 2, 0, f'{v:.2f}', ha='center', va='center', fontsize=8.5,
            color=INK if c == '#cfcec9' else 'white', weight='bold')
    ax.text(left + v / 2, .42, name, ha='center', fontsize=7.4, color=INK2)
    left += v
ax.set_xlim(0, 11.49); ax.set_ylim(-.65, .85)
ax.set_yticks([])
ax.set_xlabel('Percentage points of the 11.49-point total gain', fontsize=8.5, color=INK2)
ax.text(3.21 + 8.28 / 2, -.46, 'real cross-sectional signal: 8.28 points',
        ha='center', fontsize=7.4, color=INK2)
ax.annotate('', xy=(11.49, -.33), xytext=(3.21, -.33),
            arrowprops=dict(arrowstyle='|-|', color=MUTED, lw=1))
ax.set_title('31.0% of the real signal is look-ahead, not 71.1%',
             fontsize=9.5, color=INK, pad=6)
finish(fig, ax, 'f_attribution.png')

# =================================================== F8  connectedness heatmap
C = np.array([
 [45.8,5.5,6.8,6.1,4.2,4.6,4.1,6.1,4.0,5.7,3.5,3.5],
 [5.9,41.7,7.5,5.2,3.1,6.2,4.4,7.2,4.1,6.4,3.8,4.2],
 [6.7,6.5,43.1,5.9,3.6,5.6,4.3,7.1,3.9,5.6,3.9,3.8],
 [6.1,5.4,6.2,44.8,3.5,5.5,4.8,6.0,4.8,5.0,3.3,4.5],
 [4.4,4.1,4.4,3.7,50.0,5.5,4.4,7.0,3.7,5.5,3.7,3.6],
 [3.7,4.1,4.5,4.0,3.6,39.9,12.6,9.6,3.7,6.8,3.5,3.9],
 [4.0,3.4,4.3,3.7,3.3,14.9,40.6,7.9,3.9,7.1,3.6,3.2],
 [3.5,3.0,3.4,3.2,3.1,7.1,5.1,36.0,5.5,21.5,4.1,4.6],
 [3.1,3.4,4.1,3.5,2.9,4.5,4.2,7.3,51.3,7.1,3.9,4.8],
 [3.5,3.1,3.4,3.0,2.5,5.7,5.5,23.5,5.2,36.3,3.9,4.3],
 [2.7,2.6,3.1,2.8,3.0,4.3,4.1,5.7,3.9,5.2,55.6,7.0],
 [3.0,3.0,3.2,3.4,2.3,4.8,3.7,7.3,4.3,5.9,6.9,52.1]])
Coff = C.copy(); np.fill_diagonal(Coff, np.nan)
fig, ax = plt.subplots(figsize=(6.2, 5.0))
im = ax.imshow(Coff, cmap=SEQ, vmin=2, vmax=24)
ax.set_xticks(range(K)); ax.set_xticklabels(MKT, rotation=90, fontsize=7.5)
ax.set_yticks(range(K)); ax.set_yticklabels(MKT, fontsize=7.5)
for a in range(K):
    ax.add_patch(plt.Rectangle((a - .5, a - .5), 1, 1, fc='#f0efec', ec='white', lw=.8))
    ax.text(a, a, f'{C[a,a]:.0f}', ha='center', va='center', fontsize=6, color=MUTED)
for a, b in [(7, 9), (9, 7), (5, 6), (6, 5)]:
    ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fc='none', ec=S2, lw=1.8))
    ax.text(b, a, f'{C[a,b]:.1f}', ha='center', va='center', fontsize=6.4,
            color='white', weight='bold')
cb = fig.colorbar(im, ax=ax, fraction=.042, pad=.03)
cb.set_label('share of receiver variance (%)', fontsize=7.6, color=INK2)
cb.ax.tick_params(labelsize=7, colors=INK2)
ax.set_xlabel('transmitter  $j$', fontsize=8.5, color=INK2)
ax.set_ylabel('receiver  $i$', fontsize=8.5, color=INK2)
ax.set_title('Average connectedness, 2015–2026 (diagonal greyed)',
             fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_connectedness.png')

# =================================================== F9  clock classification
cls = ['Forward\n(admissible)', 'Backward\n(impossible)', 'Simultaneous\n(same close)',
       'Outside\n(continuous)']
mass = [137.6, 149.7, 100.9, 132.0]
pairs = [41, 41, 8, 42]
cols = [S1, S2, S4, '#cfcec9']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                               gridspec_kw={'width_ratios': [1.45, 1]})
ax1.bar(range(4), mass, color=cols, width=.62, zorder=3)
for i, (m, p) in enumerate(zip(mass, pairs)):
    ax1.text(i, m + 3, f'{m:.1f}', ha='center', fontsize=8.2, color=INK, weight='bold')
    ax1.text(i, -11, f'{p} pairs', ha='center', fontsize=7.2, color=MUTED)
ax1.set_xticks(range(4)); ax1.set_xticklabels(cls, fontsize=7.6)
ax1.set_ylabel('connectedness mass', fontsize=8.5, color=INK2)
ax1.set_ylim(-19, 172)
ax1.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
ax1.set_title('Same-day flow by physical order of closes', fontsize=9, color=INK, pad=8)

ax2.barh([0], [47.9], color=S1, height=.5, zorder=3)
ax2.barh([0], [52.1], left=[47.9], color=S2, height=.5, zorder=3)
ax2.text(47.9 / 2, 0, 'forward\n47.9%', ha='center', va='center', fontsize=8,
         color='white', weight='bold')
ax2.text(47.9 + 52.1 / 2, 0, 'backward\n52.1%', ha='center', va='center', fontsize=8,
         color='white', weight='bold')
ax2.axvline(50, color=INK, lw=1.3, ls='--')
ax2.text(50, .42, 'one half', ha='center', fontsize=7.4, color=INK)
ax2.set_xlim(0, 100); ax2.set_ylim(-.6, .7); ax2.set_yticks([])
ax2.set_xlabel('share of the 82 clock-ordered pairs (%)', fontsize=8.2, color=INK2)
ax2.set_title('What Proposition 1 predicts', fontsize=9, color=INK, pad=8)
finish(fig, [ax1, ax2], 'f_clock.png')

# =================================================== F10 learned edges network
same = [('TWSE','NKY',.776),('TWSE','KS11',.785),('TWSE','HSI',.752),('NKY','KS11',.793),
        ('NKY','HSI',.682),('NKY','DAX',.671),('KS11','HSI',.796),('SX8P','SOX',.788),
        ('SX8P','NDX',.738),('DAX','NDX',.734)]
lag = [('SOX','NKY',.783),('NDX','TWSE',.777),('NDX','NKY',.768),('UST','NKY',.757),
       ('DAX','NKY',.730),('SOX','TWSE',.727),('GLD','NKY',.713),('NDX','KS11',.695),
       ('GLD','TWSE',.684),('UST','HSI',.682)]
pos = {'TWSE':(.08,.80),'NKY':(.08,.56),'KS11':(.08,.32),'HSI':(.08,.08),
       'NIFIT':(.38,.80),'DAX':(.50,.56),'SX8P':(.50,.32),
       'NDX':(.90,.80),'REMX':(.90,.56),'SOX':(.90,.32),'GLD':(.62,.08),'UST':(.86,.08)}
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
for ax, edges_, title, col in [(axs[0], same, 'Same-day channel (forward through the clock)', S1),
                               (axs[1], lag, 'Lagged channel (back toward Asia)', S2)]:
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.06, .94); ax.axis('off')
    for s, t, w in edges_:
        p1, p2 = pos[s], pos[t]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=9,
                                     color=col, lw=.5 + (w - .65) * 9, alpha=.72,
                                     connectionstyle='arc3,rad=0.16',
                                     shrinkA=11, shrinkB=11, zorder=2))
    for m, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), .042, fc='white', ec=INK2, lw=.9, zorder=3))
        ax.text(x, y, m, ha='center', va='center', fontsize=5.9, zorder=4)
    ax.set_title(title, fontsize=9, color=INK, pad=6)
    ax.set_facecolor(SURF)
fig.patch.set_facecolor(SURF)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'f_network.png'), dpi=220, facecolor=SURF,
            bbox_inches='tight', pad_inches=.08)
plt.close(fig)
print('wrote f_network.png')

# =================================================== F11 synthetic recovery
meth = ['VAR-Granger', 'Graph learner\n(no order)', 'NOTEARS', 'Correlation',
        'Graph learner\n+ order', 'NOTEARS\n+ order']
auroc = [.541, .707, .889, .904, .909, .967]
sd = [.164, .149, .047, .035, .055, .028]
f1 = [.159, .429, .507, .511, .736, .829]
viol = [.476, .557, .462, .523, .000, .000]
ordered = [False, False, False, False, True, True]
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.3))
for ax, vals, err, ttl, ylab in [
        (axs[0], auroc, sd, 'Same-day AUROC', 'AUROC'),
        (axs[1], f1, None, 'F₁ score', 'F₁'),
        (axs[2], viol, None, 'Order violation rate', 'share of strongest edges')]:
    cols = [S1 if o else '#9ec5f4' for o in ordered]
    if ax is axs[2]:
        cols = [S3 if o else S2 for o in ordered]
    ax.bar(range(6), vals, yerr=err, color=cols, width=.64, zorder=3,
           error_kw=dict(ecolor=INK2, lw=1, capsize=2.5))
    for i, v in enumerate(vals):
        ax.text(i, v + (.035 if ax is not axs[2] else .02), f'{v:.3f}', ha='center',
                fontsize=7, color=INK2)
    ax.set_xticks(range(6)); ax.set_xticklabels(meth, fontsize=6.6, rotation=35, ha='right')
    ax.set_title(ttl, fontsize=9, color=INK, pad=6)
    ax.set_ylabel(ylab, fontsize=8, color=INK2)
    ax.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
axs[0].set_ylim(0, 1.12); axs[1].set_ylim(0, 1.0); axs[2].set_ylim(0, .68)
axs[2].axhline(.5, color=INK, ls='--', lw=1)
axs[2].text(2.5, .53, 'chance', fontsize=7, color=INK, ha='center')
finish(fig, list(axs), 'f_synthetic.png')

# =================================================== F12 robustness
corr = ['True\norder', '20%\ncorrupt', '40%\ncorrupt', '60%\ncorrupt', 'No\norder']
cA = [.909, .869, .745, .689, .707]
cS = [.055, .099, .176, .152, .149]
cV = [.000, .047, .254, .351, .557]
fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.9))
axs[0].errorbar(range(5), cA, yerr=cS, fmt='-o', color=S1, lw=2, ms=6,
                mec='white', mew=1.2, ecolor='#9ec5f4', capsize=3, zorder=3)
axs[0].axhline(.707, color=MUTED, ls='--', lw=1)
axs[0].text(2, .718, 'no-order baseline', fontsize=6.8, color=MUTED)
axs[0].set_xticks(range(5)); axs[0].set_xticklabels(corr, fontsize=7)
axs[0].set_ylabel('AUROC', fontsize=8, color=INK2)
axs[0].set_title('Degrades gracefully, never below baseline', fontsize=8.6, color=INK, pad=6)
axs[0].grid(axis='y', ls=':', color='#e3e2dd', lw=.9)

axs[1].bar(range(5), cV, color=[S3 if v == 0 else S2 for v in cV], width=.62, zorder=3)
for i, v in enumerate(cV):
    axs[1].text(i, v + .015, f'{v:.3f}', ha='center', fontsize=7, color=INK2)
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(corr, fontsize=7)
axs[1].set_ylabel('violation rate', fontsize=8, color=INK2)
axs[1].set_title('Violations stay well below unordered', fontsize=8.6, color=INK, pad=6)
axs[1].grid(axis='y', ls=':', color='#e3e2dd', lw=.9)

dens = [.15, .25, .40]; gap = [.133, .276, .308]
Ts = [1000, 2500, 4000]; gapT = [.276, .269, .267]
axs[2].plot(dens, gap, '-o', color=S1, lw=2, ms=6, mec='white', mew=1.2, label='by edge density')
ax2b = axs[2]
ax2b.plot([.15, .25, .40], gapT, '-s', color=S4, lw=2, ms=6, mec='white', mew=1.2,
          label='by sample size')
ax2b.set_xticks([.15, .25, .40])
ax2b.set_xticklabels(['0.15\nT=1000', '0.25\nT=2500', '0.40\nT=4000'], fontsize=7)
ax2b.set_ylabel('AUROC gap from the order', fontsize=8, color=INK2)
ax2b.set_ylim(.10, .35)
ax2b.legend(fontsize=7, frameon=False, loc='upper left')
ax2b.set_title('Grows with density, flat in sample size', fontsize=8.6, color=INK, pad=6)
ax2b.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
finish(fig, list(axs), 'f_robustness.png')

# =================================================== F13 linear vs nonlinear
meth3 = ['Correlation\n+ order', 'NOTEARS\n+ order', 'Graph learner\n+ order']
lin_a, non_a = [.978, .960, .882], [.995, .982, .899]
lin_f, non_f = [.812, .826, .628], [.921, .887, .668]
x = np.arange(3); w = .34
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
for ax, a, b, ttl in [(axs[0], lin_a, non_a, 'AUROC'), (axs[1], lin_f, non_f, 'F₁')]:
    ax.bar(x - w / 2, a, w, color=S1, label='Linear generator', zorder=3)
    ax.bar(x + w / 2, b, w, color=S4, label='Nonlinear (tanh)', zorder=3)
    for i in range(3):
        ax.text(i - w / 2, a[i] + .015, f'{a[i]:.3f}', ha='center', fontsize=6.8, color=INK2)
        ax.text(i + w / 2, b[i] + .015, f'{b[i]:.3f}', ha='center', fontsize=6.8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(meth3, fontsize=7.4)
    ax.set_ylim(0, 1.13); ax.set_ylabel(ttl, fontsize=8.5, color=INK2)
    ax.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
    ax.legend(fontsize=7.2, frameon=False, loc='lower left')
axs[0].set_title('The flexible model does not overtake the simple ones',
                 fontsize=9, color=INK, pad=6, loc='left')
finish(fig, list(axs), 'f_nonlinear.png')

# =================================================== F14 regime TCI
reg = ['Pre-2018\ncalm', '2018 tech\nselloff', '2019\nrecovery', 'COVID\ncrash',
       'COVID\nliquidity', '2022 rate\nshock', '2023–24\nAI rally', '2025\nonward']
tci = [54.2, 56.6, 53.9, 77.0, 59.1, 58.7, 48.2, 56.9]
sdv = [8.5, 8.3, 5.0, 8.8, 8.3, 4.7, 4.8, 7.9]
fig, ax = plt.subplots(figsize=(6.8, 3.0))
cols = [S2 if t > 70 else (S3 if t < 50 else S1) for t in tci]
ax.bar(range(8), tci, yerr=sdv, color=cols, width=.62, zorder=3,
       error_kw=dict(ecolor=INK2, lw=1, capsize=2.5))
for i, t in enumerate(tci):
    ax.text(i, t + sdv[i] + 1.4, f'{t:.1f}', ha='center', fontsize=7.6, color=INK2)
ax.set_xticks(range(8)); ax.set_xticklabels(reg, fontsize=7.2)
ax.set_ylabel('Total connectedness index (%)', fontsize=8.5, color=INK2)
ax.set_ylim(0, 92)
ax.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
ax.set_title('Markets couple under stress and decouple in calm',
             fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_regime.png')

# =================================================== F15 target autocorrelation
rho_new = [.380, .382, .558, .427, .426, .512, .336, .545, .340, .516, .287, .447]
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.bar(range(K), rho_new, color=S1, width=.62, zorder=3)
ax.axhspan(.987, .993, color=S2, alpha=.28, zorder=2)
ax.text(5.5, .955, 'a 21-day windowed estimator sits here (ρ 0.987–0.993)',
        ha='center', fontsize=7.4, color='#b8470f')
for i, v in enumerate(rho_new):
    ax.text(i, v + .022, f'{v:.3f}', ha='center', fontsize=6.8, color=INK2)
ax.set_xticks(range(K)); ax.set_xticklabels(MKT, rotation=45, fontsize=7.5, ha='right')
ax.set_ylabel('first-order autocorrelation', fontsize=8.5, color=INK2)
ax.set_ylim(0, 1.06)
ax.grid(axis='y', ls=':', color='#e3e2dd', lw=.9)
ax.set_title('The chosen target leaves something to forecast',
             fontsize=9.5, color=INK, pad=8)
finish(fig, ax, 'f_target.png')

print('all figures done')
