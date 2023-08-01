import os

data_dir1 = 'D:/Github/Learning-to-navigate-by-forgetting/data/video_dir/render_MemoNav_4goal/render_MemoNav_4goal_video_ckpt51_frame10445824.pth_Tue Sep 20 11%3A57%3A19 2022'
data_dir2 = 'D:/Github/Learning-to-navigate-by-forgetting/data/video_dir/render_exp4-11-1-4goal/render_exp4-11-1-4goal_video_ckpt51_frame10445824.pth_Tue Sep 20 12%3A06%3A49 2022'

video_list_1 = os.listdir(data_dir1)
video_list_2 = os.listdir(data_dir2)

for i in range(min(len(video_list_1), len(video_list_2))):
    if 'waypoint' in video_list_1[i]: continue

    sr1_idx = video_list_1[i].find('=')
    sr2_idx = video_list_2[i].find('=')

    sr1 = float(video_list_1[i][sr1_idx+1:sr1_idx+4])
    sr2 = float(video_list_2[i][sr2_idx+1:sr2_idx+4])

    if sr1 != 1 or sr2 != 1: continue

    spl1_idx = video_list_1[i].find('=', sr1_idx + 1)
    spl2_idx = video_list_2[i].find('=', sr2_idx + 1)

    spl1 = float(video_list_1[i][spl1_idx+1:spl1_idx+4])
    spl2 = float(video_list_2[i][spl2_idx+1:spl2_idx+4])

    if spl1 < spl2:
        print(video_list_1[i][:4], video_list_2[i][:4])

