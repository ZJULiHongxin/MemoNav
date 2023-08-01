import os, pickle, glob
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

legend_font_size = 15
tick_font_size = 10
text_font_size = 10
fig_size = (12,5)
plt.rcParams.update({'font.size': tick_font_size})
bar_width = 1
nodetype_dict = {0: 'Forgotten', 1: 'Retained'}

ckpt_lst = [''] + ['-seed{}'.format(i) for i in range(1,5)]


dist_hist = [defaultdict(dict) for _ in range(len(ckpt_lst))]

BINS = 12

dminmax = {
    'dist_to_agent': [0,BINS],
    'dist_to_goal': [0,BINS],
    'dist_to_shortestpath': [0,BINS],
    'dist2shortestpathseg_agent': [0,BINS],
    'dist2shortestpathseg_goal': [0,BINS],
}

legend_lst = ['agent', 'goal', 'shortest path', 'shortest path near agent', 'shortest path near goal']

dist_bins = np.linspace(dminmax['dist_to_agent'][0], dminmax['dist_to_agent'][1], BINS + 1)
entries = list(dminmax.keys())


# For detailed statstices
dist_hist = [defaultdict(dict) for _ in range(len(ckpt_lst))]

for g in range(3):
    print('Goal',g)
    for i, ckpt in enumerate(ckpt_lst):
        eval_result_dir = './eval_results/exp4-11{}-3goal-dist/*.pkl'.format(ckpt)

        pkl_file = glob.glob(eval_result_dir)

        with open(pkl_file[0], 'rb') as f:
            dist_info = pickle.load(f)

        scene_diagonal = dist_info['scene_diagonal']
    
        for k in entries:
            coef = 1.5 if 'goal' not in k else 2.5
            for j in range(2):
                dist2entry = dist_info[k][g][j]
                dist_hist[i][k][j] = np.zeros(shape=(BINS))

                # for bin_id in range(dist_bins.shape[0] - 1):
                #     idxs = np.where(np.logical_and(dist2entry >= dist_bins[bin_id], dist2entry < dist_bins[bin_id+1]))
                #     dist_hist[i][k][j][bin_id] = idxs[0].shape[0]
                
                for t, v in enumerate(dist2entry):
                    if v > dminmax[k][1]: continue
                    bin_id = np.nonzero(v >= dist_bins)[0][-1]

                    
                    if True:#not (j == 0 and v < 8 and np.random.random() > coef * v / BINS):
                        dist_hist[i][k][j][bin_id] += 1
                    #dist_hist[i][k][j][bin_id] += 1
                    # if t >= 50: break

                dist_hist[i][k][j] /= dist_hist[i][k][j].sum()

    radius = 2 # 
    for j in range(2):
        print(nodetype_dict[j])
        for k in entries:
            # print(dist_hist[i][k][j])
            temp = np.array([dist_hist[i][k][j][:radius].sum() for i in range(len(ckpt_lst))])
            print(nodetype_dict[j] + ' nodes to ' + k, 'mean: {:.3f}%, std: {:.3f}%'.format(temp.mean(), temp.std()))

        print()
    
    # For drawing histograms
    for i, ckpt in enumerate(ckpt_lst):
        eval_result_dir = './eval_results/exp4-11{}-3goal-dist/*.pkl'.format(ckpt)

        pkl_file = glob.glob(eval_result_dir)

        with open(pkl_file[0], 'rb') as f:
            dist_info = pickle.load(f)

        scene_diagonal = dist_info['scene_diagonal']

        # print(sum(dist_info['dist_to_goal'][0]), sum(dist_info['dist_to_agent'][0])); input()
        for k in entries:
            coef = 1.5 if 'goal' not in k else 2.5
            for j in range(2):
                dist2entry = dist_info[k][g][j]
                dist_hist[i][k][j] = []
                for idx in range(len(dist2entry)):
                    if dist2entry[idx] > 25: continue
                    #dist_hist[i][k][j].append(dist2entry[idx])
                    if True:#not (j == 0 and dist2entry[idx] < 8 and np.random.random() > coef * dist2entry[idx] / BINS):
                        dist_hist[i][k][j].append(dist2entry[idx])


    ckpt_aggre = defaultdict(dict)
    mean_each_ckpt = defaultdict(dict)

    for j in range(2):
        for k in entries:
            aggre = np.concatenate([dist_hist[i][k][j] for i in range(len(ckpt_lst))])
            ckpt_aggre[k][j] = aggre

            mean_each_ckpt[k][j] = np.array([np.mean(dist_hist[i][k][j]) for i in range(len(ckpt_lst))])



    f = plt.figure(figsize=fig_size)


    for i in range(2):
        print(nodetype_dict[i])
        for j, k in enumerate(entries):
            ax = plt.subplot(2,len(entries),len(entries)*i+j+1)
            ax.hist(ckpt_aggre[k][i], bins=dist_bins, density=True, color='g' if i==0 else 'orange')
            #plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')
            ax.set_xlabel('Distance [m]', fontsize=tick_font_size+3)
            ax.set_ylabel('Proportion', fontsize=tick_font_size+3)
            plt.xticks(np.arange(0,BINS+1), [int(x) for x in dist_bins], fontsize=tick_font_size)
            ax.set_ylim((0.0,0.28))

            title = 'Dist. to ' + legend_lst[j] # nodetype_dict[i] + ' nodes to ' + legend_lst[j]
            ax.set_title(title)

            print(title, '{:.2f} ({:.2f})m'.format(np.mean(mean_each_ckpt[k][i]), np.std(mean_each_ckpt[k][i])))

    print()
# f = plt.figure()

# nodetype_dict = {0: 'forgotten', 1: 'retained'}
# for i in range(2):
#     for j, k in enumerate(entries):
#         ax = plt.subplot(2,3,3*i+j+1)
#         X = np.linspace(dminmax[k][0], dminmax[k][1], BINS + 1)[:-1].astype(int)
#         Y = avg_std[k][i][0]
#         plt.bar(X, Y, width=bar_width, color='g' if i == 0 else 'orange')
#         plt.xlabel('Distance [m]')
#         plt.ylabel('Proportion')
#         plt.xticks(X, X, fontsize=tick_font_size)
#         ax.set_title(nodetype_dict[i] + '-' + k)


#total_prop_within_radius = 
    plt.show()