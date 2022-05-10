import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import imageio
from copy import deepcopy
import json
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-episodes", type=int, default=1007)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--gpu", type=str, default="0,0", help="Simulation and evaluation GPU IDs")
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--stop", action='store_true', default=False)
parser.add_argument("--forget", action='store_true', default=False)
parser.add_argument("--diff", choices=['random', 'easy', 'medium', 'hard'], default='')
parser.add_argument("--split", choices=['val', 'train', 'min_val'], default='val')
parser.add_argument('--eval-ckpt', type=str, required=True)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--record', choices=['0','1','2','3'], default='0') # 0: no record 1: env.render 2: pose + action numerical traj 3: features
parser.add_argument('--th', type=str, default='0.75') # s_th
parser.add_argument('--record-dir', type=str, default='data/video_dir')

args = parser.parse_args()
args.record = int(args.record)
args.th = float(args.th)
import os
import time, datetime
import cv2
os.environ['GLOG_minloglevel'] = "3"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"

import numpy as np
import torch
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enable = True
torch.set_num_threads(5)
from env_utils.make_env_utils import add_panoramic_camera
import habitat
from habitat import make_dataset
from env_utils.task_search_env import SearchEnv
from configs.default import get_config, CN
from runner import *


def eval_config(args):
    config = get_config(args.config, args.version, create_folders=False)
    config.defrost()
    config.use_depth = config.TASK_CONFIG.use_depth = True
    config.DIFFICULTY = args.diff
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG, normalize_depth=True)
    config.TASK_CONFIG.DATASET.SPLIT = args.split if 'gibson' in config.TASK_CONFIG.DATASET.DATA_PATH else 'test'
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    if 'COLLISIONS' not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS += ['COLLISIONS']
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    if config.TASK_CONFIG.DATASET.CONTENT_SCENES == ['*']:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    else:
        scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = scenes
    ep_per_env = int(np.ceil(args.num_episodes / len(scenes)))
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = ep_per_env
    if args.stop:
        config.ACTION_DIM = 4
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS= ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    else:
        config.ACTION_DIM = 3
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        config.TASK_CONFIG.TASK.SUCCESS.TYPE = "Success_woSTOP"

    if args.forget:
        config.memory.FORGET= True
    config.TASK_CONFIG.TASK.MEASUREMENTS.insert(0,'GOAL_INDEX')

    config.freeze()
    return config

def load(ckpt):
    new_state_dict, env_global_node = None, None
    if ckpt != 'none':
        sd = torch.load(ckpt,map_location=torch.device('cpu'))
        state_dict = sd['state_dict']
        new_state_dict = {}

        env_global_node = sd.get("env_global_node", None)
        ckpt_config = sd.get("config", None)

        for key in state_dict.keys():
            if 'actor_critic' in key:
                new_state_dict[key[len('actor_critic.'):]] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        
        return (new_state_dict, env_global_node, ckpt_config)


#TODO: ADD runner in the config file e.g. config.runner = 'VGMRunner' or 'BaseRunner'
def evaluate(eval_config, ckpt):
    if args.record > 0:
        if not os.path.exists(os.path.join(args.record_dir, eval_config.VERSION)):
            os.mkdir(os.path.join(args.record_dir, eval_config.VERSION))

        VIDEO_DIR = os.path.join(args.record_dir, eval_config.VERSION + '_video_' + ckpt.split('/')[-1] + '_' +str(time.ctime()))
        if not os.path.exists(VIDEO_DIR): os.mkdir(VIDEO_DIR)

    state_dict, env_global_node, ckpt_config = load(ckpt)


    if ckpt_config is not None:
        task_config = eval_config.TASK_CONFIG
        ckpt_config.defrost()
        task_config.defrost()
        ckpt_config.TASK_CONFIG = task_config
        ckpt_config.runner = eval_config.runner
        ckpt_config.AGENT_TASK = 'search'
        ckpt_config.DIFFICULTY = eval_config.DIFFICULTY
        ckpt_config.ACTION_DIM = eval_config.ACTION_DIM
        ckpt_config.memory = eval_config.memory
        ckpt_config.scene_data = eval_config.scene_data
        ckpt_config.WRAPPER = eval_config.WRAPPER
        ckpt_config.REWARD_METHOD = eval_config.REWARD_METHOD
        ckpt_config.ENV_NAME = eval_config.ENV_NAME
        ckpt_config.VERSION = eval_config.VERSION
        ckpt_config.POLICY = eval_config.POLICY
        ckpt_config.GCN = eval_config.GCN

        for k, v in eval_config.items():
            if k not in ckpt_config:
                ckpt_config.update({k:v})
            if isinstance(v, CN):
                for kk, vv in v.items():
                    if kk not in ckpt_config[k]:
                        ckpt_config[k].update({kk: vv})
        
        ckpt_config.update({"SIMULATOR_GPU_ID": args.gpu[0]})
        ckpt_config.update({"TORCH_GPU_ID": args.gpu[-1]})

        ckpt_config.freeze()
        eval_config = ckpt_config

    eval_config.defrost()
    eval_config.th = args.th

    eval_config.record = False # record from this side , not in env
    eval_config.render_map = args.record > 0 or args.render or 'hand' in args.config
    eval_config.noisy_actuation = True
    eval_config.freeze()
    # VGMRunner
    runner = eval(eval_config.runner)(eval_config, env_global_node=env_global_node, return_features=True)

    eval_info = ''
    eval_info += '=========================================\n'
    eval_info += 'Version Name: {}\n'.format(eval_config.VERSION)
    eval_info += 'Task config path: {}\n'.format(eval_config.BASE_TASK_CONFIG_PATH)
    eval_info += 'Runner: {}\n'.format(eval_config.runner)
    eval_info += 'Policy: {}\n'.format(eval_config.POLICY)
    eval_info += 'Num params: {}\n'.format(sum(param.numel() for param in runner.parameters()))
    eval_info += 'Difficulty: {}\n'.format(eval_config.DIFFICULTY)
    eval_info += 'Stop action: {}\n'.format('True' if eval_config.ACTION_DIM==4 else 'False')
    eval_info += 'Env gloabl node: {}, link percentage: {}, random_replace: {}\n'.format(str(eval_config.GCN.WITH_ENV_GLOBAL_NODE), str(eval_config.GCN.ENV_GLOBAL_NODE_LINK_RANGE), str(eval_config.GCN.RANDOM_REPLACE))

    if eval_config.memory.FORGET:
        num_forgotten_nodes = "{}%".format(int(100 * eval_config.memory.RANK_THRESHOLD)) if eval_config.memory.RANK_THRESHOLD < 1 else "{}".format(int(eval_config.memory.RANK_THRESHOLD))
        if eval_config.memory.RANK == 'bottom':
            eval_info += 'Forgetting: {} \n\t Start forgetting after {} nodes are collected\n\t Nodes in the bottom {} will be forgotten\n'.format(str(eval_config.memory.FORGET), eval_config.memory.TOLERANCE, num_forgotten_nodes)
        elif eval_config.memory.RANK == 'top':
            eval_info += 'Forgetting: {} \n\t Start forgetting after {} nodes are collected\n\t Nodes in the top {} will be remembered\n'.format(str(eval_config.memory.FORGET), eval_config.memory.TOLERANCE, num_forgotten_nodes)
        eval_info += '\t Forgetting according to {} attention scores\n'.format(eval_config.memory.FORGETTING_ATTN)
    else:
        eval_info += 'Forgetting: False\n'
    eval_info += 'GCN encoder type: {}\n'.format(eval_config.GCN.TYPE)
    eval_info += 'Fusion method: {}, decode global node: {}\n'.format(str(eval_config.FUSION_TYPE), str(eval_config.transformer.DECODE_GLOBAL_NODE))
    eval_info += '===========================================\n'
    
    print(eval_info)

    runner.eval()
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda:"+str(eval_config.TORCH_GPU_ID))
        runner.to(device)
    #runner.load(state_dict)

    try:
        runner.load(state_dict)
    except:
        raise
        agent_dict = runner.agent.state_dict()
        new_sd = {k: v for k, v in state_dict.items() if k in agent_dict.keys() and (v.shape == agent_dict[k].shape)}
        agent_dict.update(new_sd)
        runner.load(agent_dict)

    env = eval(eval_config.ENV_NAME)(eval_config)
    env.habitat_env._sim.seed(args.seed)
    if runner.need_env_wrapper:
        env = runner.wrap_env(env,eval_config)

    result = {}
    result['config'] = eval_config
    result['args'] = args
    result['version'] = args.version
    result['start_time'] = time.ctime()
    result['noisy_action'] = env.noise
    scene_dict = {}
    render_check = False
    with torch.no_grad():
        ep_list = []
        total_success, total_spl,  total_success_timesteps = [], [], []
        
        attn_choice = "goal_attn"
        if "cur" in eval_config.memory.FORGETTING_ATTN.lower():
            attn_choice = "curr_attn"
        elif "global" in eval_config.memory.FORGETTING_ATTN.lower() or "gat" in eval_config.memory.FORGETTING_ATTN.lower():
            attn_choice = "GAT_attn"
        
        temp_s = "\ntest on {} episodes\n".format(args.num_episodes)
        print(temp_s)
        eval_info += temp_s
        
        for episode_id in range(args.num_episodes):
            obs = env.reset()
            if render_check == False:
                if obs['panoramic_rgb'].sum() == 0 :
                    print('NO RENDERING!!!!!!!!!!!!!!!!!! YOU SHOULD CHECK YOUT DISPLAY SETTING')
                else:
                    render_check=True
            runner.reset()
            scene_name = env.current_episode.scene_id.split('/')[-1][:-4]
            if scene_name not in scene_dict.keys():
                scene_dict[scene_name] = {'success': [], 'spl': [], 'avg_step': [], 'avg_node_num': [0,0]}
            done = True
            reward = None
            info = None
            if args.record > 0:
                img = env.render('rgb')
                imgs = [img]
            step = 0
            while True:
                action, att_scores, decision_time = runner.step(obs, reward, done, info, env)
                
                env.forget_node(
                    att_scores[attn_choice],
                    num_nodes=obs['global_mask'].sum(dim=1),
                    att_type=attn_choice)

                obs, reward, done, info = env.step(action)
                step += 1
                if args.record > 0:
                    img = env.render('rgb')
                    imgs.append(img)
                if args.render:
                    env.render('human')
                if done: break
            
            spl = info['spl']
            if np.isnan(spl):
                spl = 0.0
            scene_dict[scene_name]['success'].append(info['success'])
            scene_dict[scene_name]['spl'].append(spl)
            total_success.append(info['success'])
            total_spl.append(spl)

            if info['success']:
                scene_dict[scene_name]['avg_step'].append(step)
                total_success_timesteps.append(step)

            ep_list.append({'house': scene_name,
                            'ep_id': env.current_episode.episode_id,
                            'start_pose': [env.current_episode.start_position, env.current_episode.start_rotation],
                            'target_pose': env.habitat_env.task.sensor_suite.sensors['target_goal'].goal_pose,
                            'total_step': step,
                            'collision': info['collisions']['count'] if isinstance(info['collisions'], dict) else info['collisions'],
                            'success': info['success'],
                            'spl': spl,
                            'distance_to_goal': info['distance_to_goal'],
                            'target_distance': env.habitat_env._sim.geodesic_distance(env.habitat_env.current_episode.goals[0].position,env.current_episode.start_position)})
            if args.record > 0:
                video_name = os.path.join(VIDEO_DIR,'%04d_%s_success=%.1f_spl=%.1f.mp4'%(episode_id, scene_name, info['success'], spl))
                with imageio.get_writer(video_name, fps=30) as writer:
                    im_shape = imgs[-1].shape
                    for im in imgs:
                        if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
                            im = cv2.resize(im, (im_shape[1], im_shape[0]))
                        writer.append_data(im.astype(np.uint8))
                    writer.close()
            print('[%04d/%04d] %s success %.4f, spl %.4f, steps %.4f || total success %.4f, spl %.4f, success time step %.2f' % (episode_id,
                                                          args.num_episodes,
                                                          scene_name,
                                                          info['success'],
                                                          spl,
                                                          step,
                                                          np.array(total_success).mean(),
                                                          np.array(total_spl).mean(),
                                                          np.array(total_success_timesteps).mean(),
                                                          ))
    
    result['eval_info'] = eval_info
    result['detailed_info'] = ep_list
    result['each_house_result'] = {}

    success = []
    spl = []

    for scene_name in scene_dict.keys():
        mean_success = np.array(scene_dict[scene_name]['success']).mean().item()
        mean_spl = np.array(scene_dict[scene_name]['spl']).mean().item()
        mean_step = np.array(scene_dict[scene_name]['avg_step']).mean().item()

        result['each_house_result'][scene_name] = {'success': mean_success, 'spl': mean_spl, 'avg_step': mean_step}
        print('SCENE %s: success %.4f, spl %.4f, avg steps %.2f'%(scene_name, mean_success,mean_spl, mean_step))
        success.extend(scene_dict[scene_name]['success'])
        spl.extend(scene_dict[scene_name]['spl'])
    
    result['avg_success'] = np.array(success).mean().item()
    result['avg_spl'] = np.array(spl).mean().item()
    result['avg_timesteps'] = np.array(total_success_timesteps).mean().item()
    print('================================================')
    print('avg success : %.4f'%result['avg_success'])
    print('avg spl : %.4f'%result['avg_spl'])
    print('avg timesteps : %.4f'% result['avg_timesteps'])
    env.close()
    return result

if __name__=='__main__':
    import joblib
    import glob
    cfg = eval_config(args)
    if os.path.isdir(args.eval_ckpt):
        print('eval_ckpt ', args.eval_ckpt, ' is directory')
        ckpts = [os.path.join(args.eval_ckpt,x) for x in sorted(os.listdir(args.eval_ckpt))]
        ckpts.reverse()
    elif os.path.exists(args.eval_ckpt):
        ckpts = args.eval_ckpt.split(",")
    else:
        ckpts = [x for x in sorted(glob.glob(args.eval_ckpt+'*'))]
        ckpts.reverse()   
    print('evaluate total {} ckpts'.format(len(ckpts)))
    print(ckpts)

    eval_results_dir = "eval_results"
    if not os.path.exists(eval_results_dir):
        os.mkdir(eval_results_dir)
    
    this_exp_dir = os.path.join(eval_results_dir, cfg.VERSION)
    if not os.path.exists(this_exp_dir):
        os.mkdir(this_exp_dir)
    
    for ckpt in ckpts:
        if 'ipynb' in ckpt or 'pt' not in ckpt: continue
        print('============================', ckpt.split('/')[-1], '==================')
        result = evaluate(cfg, ckpt)

        ckpt_name = ckpt.split('/')[-1].replace('.','').replace('pth','')
        
        each_scene_results_txt = os.path.join(this_exp_dir, "{}_{}_{}.txt".format(cfg.VERSION, ckpt_name, args.diff))
        with open(each_scene_results_txt, 'w') as f:
            f.write(result['eval_info'])

            lines = ["Avg SR: {:.4f}, Avg SPL: {:.4f}, Avg success timestep: {:.1f}\n".format(result['avg_success'], result['avg_spl'], result['avg_timesteps'])]
            lines.append("Results of each scene:\n")
            for k, v in result['each_house_result'].items():
                this_scene = result['each_house_result'][k]
                lines.append("{}: SR {:.4f}, SPL {:.4f}, Avg step {:.2f}\n".format(k, this_scene['success'], this_scene['spl'], this_scene['avg_step']))
 
            f.writelines(lines)
        print("save brief eval results to", each_scene_results_txt)

        # detailed eval results
        eval_data_name = os.path.join(this_exp_dir, '{}_{}_{}.dat.gz'.format(cfg.VERSION, ckpt_name, args.diff))
        if os.path.exists(eval_data_name):
            data = joblib.load(eval_data_name)
            if cfg.VERSION in data.keys():
                data[cfg.VERSION].update({ckpt + '_{}'.format(time.time()): result})
            else:
                data.update({cfg.VERSION: {ckpt + '_{}'.format(time.time()): result}})
        else:
            data = {cfg.VERSION: {ckpt + '_{}'.format(time.time()): result}}
        joblib.dump(data, eval_data_name)

        print("save detailed eval results to", eval_data_name)

        t = datetime.datetime.now()
        print('Evaluation completed: ', datetime.datetime.strftime(t,'%Y-%m-%d %H:%M:%S'))
