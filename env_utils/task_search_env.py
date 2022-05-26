from ipaddress import collapse_addresses
from operator import mod
from tkinter.messagebox import NO
from typing import Optional, Type
from habitat import Config, Dataset
import cv2
import matplotlib
import matplotlib.pyplot as plt
from utils.vis_utils import observations_to_image, append_text_to_image
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from habitat.core.spaces import ActionSpace, EmptySpace
import numpy as np
from env_utils.custom_habitat_env import RLEnv, MIN_DIST, MAX_DIST
import habitat
from habitat.utils.visualizations.utils import images_to_video
from habitat.utils.visualizations import maps, utils

from env_utils.custom_habitat_map import TopDownGraphMap
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import env_utils.noisy_actions
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs

import imageio
import os
import time
import pickle
import quaternion as q
import scipy
from habitat.tasks.utils import cartesian_to_polar, quaternion_to_rotation
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
import torch

class SearchEnv(RLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self.noise = config.noisy_actuation
        self.record = config.record
        self.render_map = getattr(config,'render_map',False) or self.record
        self.record_interval = config.VIS_INTERVAL
        self.record_dir = config.VIDEO_DIR
        task_config = config.TASK_CONFIG
        
        task_config.defrost()
        task_config.TASK.SUCCESS.SUCCESS_DISTANCE = config.RL.SUCCESS_DISTANCE
        if self.render_map:
            task_config.TASK.TOP_DOWN_GRAPH_MAP = config.TASK_CONFIG.TASK.TOP_DOWN_MAP.clone()
            task_config.TASK.TOP_DOWN_GRAPH_MAP.TYPE = "TopDownGraphMap"
            task_config.TASK.TOP_DOWN_GRAPH_MAP.MAP_RESOLUTION = 1024
            task_config.TASK.TOP_DOWN_GRAPH_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 10000
            if getattr(config, 'map_more', False):
                task_config.TASK.TOP_DOWN_GRAPH_MAP.MAP_RESOLUTION = 2500
                task_config.TASK.TOP_DOWN_GRAPH_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 10000
            task_config.TASK.TOP_DOWN_GRAPH_MAP.DRAW_CURR_LOCATION = getattr(config, 'GRAPH_LOCATION', 'point')
            task_config.TASK.MEASUREMENTS += ['TOP_DOWN_GRAPH_MAP']
            
        if 'TOP_DOWN_MAP' in config.TASK_CONFIG.TASK.MEASUREMENTS:
            task_config.TASK.MEASUREMENTS = [k for k in task_config.TASK.MEASUREMENTS if
                                                    'TOP_DOWN_MAP' != k]
        task_config.SIMULATOR.ACTION_SPACE_CONFIG = "CustomActionSpaceConfiguration"
        task_config.TASK.POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS + ['NOISY_FORWARD', 'NOISY_RIGHT', 'NOISY_LEFT']
        task_config.TASK.ACTIONS.NOISY_FORWARD = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_FORWARD.TYPE = "NOISYFORWARD"
        task_config.TASK.ACTIONS.NOISY_RIGHT = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_RIGHT.TYPE = "NOISYRIGHT"
        task_config.TASK.ACTIONS.NOISY_LEFT = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_LEFT.TYPE = "NOISYLEFT"
        task_config.freeze()
        self.config = config
        self.task_config = task_config
        self._core_env_config = config.TASK_CONFIG
        self.success_distance = config.RL.SUCCESS_DISTANCE
        self._previous_measure = None
        self._previous_action = -1
        self.time_t = 0
        self.stuck = 0
        self.follower = None
        if 'NOISY_FORWARD' not in HabitatSimActions:
            HabitatSimActions.extend_action_space("NOISY_FORWARD")
            HabitatSimActions.extend_action_space("NOISY_RIGHT")
            HabitatSimActions.extend_action_space("NOISY_LEFT")

        if self.noise: moves = ["NOISY_FORWARD", "NOISY_LEFT", "NOISY_RIGHT"]
        else: moves = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        if 'STOP' in task_config.TASK.POSSIBLE_ACTIONS:
            self.action_dict = {id+1: move for id, move in enumerate(moves)}
            self.action_dict[0] = "STOP"
        else:
            self.action_dict = {id: move for id, move in enumerate(moves)}

        self.SUCCESS_REWARD = self.config.RL.SUCCESS_REWARD
        self.COLLISION_REWARD = self.config.RL.COLLISION_REWARD
        self.SLACK_REWARD = self.config.RL.SLACK_REWARD

        # Segmentation fault (core dumped) occurred here
        super().__init__(self._core_env_config, dataset)

        act_dict = {"MOVE_FORWARD": EmptySpace(),
                    'TURN_LEFT': EmptySpace(),
                    'TURN_RIGHT': EmptySpace()
        }
        if 'STOP' in task_config.TASK.POSSIBLE_ACTIONS:
            act_dict.update({'STOP': EmptySpace()})
        self.action_space = ActionSpace(act_dict)


        """
        self.habitat_env is class Env in custom_habitat_env.py
        """
        obs_dict = {
                'panoramic_rgb': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_rgb'],
                'panoramic_depth': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_depth'],
                'target_goal': self.habitat_env._task.sensor_suite.observation_spaces.spaces['target_goal'],
                'step': Box(low=np.array(0),high=np.array(500), dtype=np.float32),
                'prev_act': Box(low=np.array(-1), high=np.array(self.action_space.n), dtype=np.int32),
                'gt_action': Box(low=np.array(-1), high=np.array(self.action_space.n), dtype=np.int32),
                'target_pose': Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32),
                'distance': Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            }
        
        self.mapper = self.habitat_env.task.measurements.measures['top_down_map'] if self.render_map else None

        if self.config.USE_AUXILIARY_INFO:
            self.return_have_been = True
            self.return_target_dist_score = True
            obs_dict.update({
                            'have_been': Box(low=0, high=1, shape=(1,), dtype=np.int32),
                             'target_dist_score': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                             })
        else:
            self.return_have_been = False
            self.return_target_dist_score = False
        self.observation_space = SpaceDict(obs_dict)

        self.habitat_env.difficulty = config.DIFFICULTY
        if config.DIFFICULTY == 'easy':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 1.5, 3.0
        elif config.DIFFICULTY == 'medium':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 3.0, 5.0
        elif config.DIFFICULTY == 'hard':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 5.0, 10.0
        elif config.DIFFICULTY == 'random':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 1.5, 10.0
        elif config.DIFFICULTY == '2goal':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 10.0, 15.0
        elif config.DIFFICULTY == '3goal':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 15.0, 25.0
        elif config.DIFFICULTY == '4goal':
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 20.0, 40.0
        else:
            raise NotImplementedError
        print('[SearchEnv] Current difficulty %s, MIN_DIST %f, MAX_DIST %f - # goals %d'%(config.DIFFICULTY, self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST, self.habitat_env._num_goals))

        self.get_reward = self.get_progress_reward

        self.number_of_episodes = 1000
        self.has_log_info = None

        self.done_type = 0

    def update_graph(self, node_list, affinity, graph_mask, curr_info, flags=None):
        if self.mapper is not None and self.render_map:
            self.mapper.update_graph(node_list, affinity, graph_mask, curr_info, flags)

    def build_path_follower(self, each_goal=False):
        self.follower = ShortestPathFollower(self._env.sim, self.success_distance, False)\

    def get_best_action(self, goal=None):
        if self.follower is None:
            self.build_path_follower()
        curr_goal = goal if goal is not None else self.curr_goal.position
        act = self.follower.get_next_action(curr_goal)
        return act

    def set_random_goals(self):
        pose = self.current_position
        try_num = 0
        while True:
            goal_pose = self.habitat_env._sim.sample_navigable_point()
            same_floor = abs(goal_pose[1] - self.initial_pose[1]) < 0.05
            far = self.habitat_env._sim.geodesic_distance(pose, goal_pose) > 7.
            if same_floor and far:
                self.random_goals = goal_pose
                break
            try_num += 1
            if try_num > 100 and same_floor :
                self.random_goals = goal_pose
                break

    def get_random_goal_action(self):
        return self.follower.get_next_action(self.random_goals)

    def get_dist(self, goal_position):
        return self.habitat_env._sim.geodesic_distance(self.current_position, goal_position)

    @property
    def recording_now(self):
        return self.record and self.habitat_env._total_episode_id%self.record_interval == 0

    @property
    def curr_goal_idx(self): return 0

    @property
    def curr_goal(self):
        return self.current_episode.goals[min(self.curr_goal_idx, len(self.current_episode.goals)-1)]

    def reset(self):
        self._previous_action = -1
        self.time_t = 0
        observations = super().reset()

        self.num_goals = len(self.current_episode.goals)
        self._previous_measure = self.get_dist(self.curr_goal.position)
        self.initial_pose = self.current_position

        self.info = None
        self.total_reward = 0
        self.progress = 0
        self.stuck = 0
        self.min_measure = self.habitat_env.MAX_DIST
        self.prev_coverage = 0
        self.prev_position = self.current_position.copy()
        self.prev_rotation = self.current_rotation.copy()
        self.positions = [self.current_position]
        self.obs = self.process_obs(observations)
        self.has_log_info = None
        self.imgs = []
        if self.recording_now:
            self.imgs.append(self.render('rgb'))
        return self.obs

    @property
    def scene_name(self): # version compatibility
        if hasattr(self.habitat_env._sim, 'habitat_config'):
            sim_scene = self.habitat_env._sim.habitat_config.SCENE
        else:
            sim_scene = self.habitat_env._sim.config.SCENE
        return sim_scene

    def process_obs(self, obs):
        obs['episode_id'] = self.current_episode.episode_id
        obs['step'] = self.time_t
        obs['position'] = self.current_position # habitat_env.sim.get_agent_state().position
        obs['rotation'] = self.current_rotation.components
        obs['target_pose'] = self.curr_goal.position
        obs['distance'] = self.get_dist(obs['target_pose'])
        # obs['target_goal'] = obs['target_goal'][0] if self.num_goals == 1 else obs['target_goal']

        # When the agent finishes navigating to the final goal, self.curr_goal_idx will exceed the maximum index of goals.
        # So we take minimum here to avoid index out of range
        obs['target_goal'] = obs['target_goal'][self.curr_goal_idx]
        if self.return_have_been:
            if len(self.positions) < 10:
                have_been = False
            else:
                dists = np.linalg.norm(np.array(self.positions) - np.expand_dims(np.array(self.current_position),0), axis=1)
                far = np.where(dists > 1.0)[0]
                near = np.where(dists[:-10] < 1.0)[0]
                have_been = len(far) > 0 and len(near) > 0 and (near < far.max()).any()
            obs['have_been'] = np.array(have_been).astype(np.float32).reshape(1)

        if self.return_target_dist_score:
            target_dist_score = np.maximum(1-np.array(obs['distance'])/2.,0.0)
            obs['target_dist_score'] = np.array(target_dist_score).astype(np.float32).reshape(1)
        return obs

    def step(self, action):
        if isinstance(action, dict):
            action = action['action']
        self._previous_action = action

        obs, reward, done, self.info = super().step(self.action_dict[action]) # super().step() is defined in line 469 in custom_habitat_env.py

        self.time_t += 1
        self.info['length'] = self.time_t * done
        self.info['episode'] = int(self.current_episode.episode_id)
        self.info['distance_to_goal'] = self._previous_measure
        self.info['step'] = self.time_t
        self.positions.append(self.current_position)
        self.obs = self.process_obs(obs)
        self.total_reward += reward
        self.prev_position = self.current_position.copy()
        self.prev_rotation = self.current_rotation.copy()
        if self.recording_now:
            self.imgs.append(self.render('rgb'))
            if done: self.save_video(self.imgs)
        return self.obs, reward, done, self.info

    def save_video(self,imgs):
        video_name = 'ep_%03d_scene_%s.mp4'%(self.habitat_env._total_episode_id, self.scene_name.split('/')[-1][:-4])
        w, h = imgs[0].shape[0:2]
        resize_h, resize_w = (h//16)*16, (w//16)*16
        imgs = [cv2.resize(img, dsize=(resize_h, resize_w)) for img in imgs]
        imageio.mimsave(os.path.join(self.record_dir, video_name), imgs, fps=40)

    def get_reward_range(self):
        return (
            self.SLACK_REWARD - 1.0,
            self.SUCCESS_REWARD + 1.0,
        )

    def get_progress_reward(self, observations):
        reward = self.SLACK_REWARD
        current_measure = self.get_dist(self.curr_goal.position)
        self.move = self._previous_measure - current_measure
        reward += max(self.move,0.0) * 0.2
        if abs(self.move) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0
        self._previous_measure = current_measure
        if self._episode_success():
            reward += self.SUCCESS_REWARD * self._env.get_metrics()['spl']
        return reward

    def _episode_success(self):
        return self._env.get_metrics()['success']

    def get_success(self):
        return self._episode_success()

    def get_done(self, observations):
        done, self.done_type = False, 0
        if self._env.episode_over or self._episode_success():
            done = True
        pose = self.current_position
        diff_floor = abs(pose[1] - self.initial_pose[1]) > 0.5
        if self.stuck > 100 or diff_floor:
            done = True
            self.done_type = 1
        if np.isinf(self._env.get_metrics()['distance_to_goal']):
            done = True
            self.done_type = 2
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["done_type"] = self.done_type
        info["change_goal"] = False
        return info

    @property
    def current_position(self):
        return self.habitat_env.sim.get_agent_state().position
    @property
    def current_rotation(self):
        return self.habitat_env.sim.get_agent_state().rotation

    def get_episode_over(self):
        return self._env.episode_over

    def get_agent_state(self):
        return self.habitat_env.sim.get_agent_state()

    def get_curr_goal_index(self):
        return self.curr_goal_idx

    def log_info(self, log_type='str', info=None):
        self.has_log_info = {'type': log_type,
                             'info': info}

    def convert_points_to_topdown(self, pathfinder, points, meters_per_pixel):
        points_topdown = []
        bounds = pathfinder.get_bounds()
        for point in points:
            # convert 3D x,z to topdown x,y
            v = (point[0] - bounds[0][0]) / meters_per_pixel
            u = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([int(u), int(v)]))
        return points_topdown
    
    def init_map_settings(self):
        if not self.render_map:return
        self.map_res = self.task_config.TASK.TOP_DOWN_GRAPH_MAP.MAP_RESOLUTION // 4
        self.meters_per_pixel = maps.calculate_meters_per_pixel(self.map_res, self.habitat_env.sim, self.habitat_env.sim.pathfinder) 
        self.recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        node_side = self.map_res // 32
        self.node_side = node_side
        self.square = np.tile(np.array([[[255,0,0]]], dtype=np.uint8), (node_side, node_side, 1))
        self.target_square = np.tile(np.array([[[0,240,0]]], dtype=np.uint8), (node_side, node_side, 1)) 
        self.cmap = matplotlib.cm.get_cmap("rainbow")
    #def draw_traj(self, map,):

    def render1(self, mode='rgb', waypoint_pose=None, att_features=None, forget_node_indices=None, imshow=False):
        # waypoint_pose: a list of waypoint xyz coords
        # att_features: the att scores of the last GATv2 layer; size: num_nodes
        # forget_node_indices: 
        info = self.get_info(None) if self.info is None else self.info # ['distance_to_goal', 'success', 'spl', 'collisions', 'top_down_map']
        
        # img = info["top_down_map"]["map"]
        # img = maps.colorize_topdown_map(
        #         img, info["top_down_map"]["fog_of_war_mask"]
        #     )
        
        top_down_map = None
        if att_features is not None:
            top_down_map = maps.get_topdown_map(self.habitat_env.sim.pathfinder, height=waypoint_pose[0][1], meters_per_pixel=self.meters_per_pixel) # uint8 numpy array
            top_down_map = self.recolor_map[top_down_map]

             # create a colored square of size side x side x 3
            waypoint_pose.append(self.obs['target_pose'])
            xy_vis_points = self.convert_points_to_topdown(self.habitat_env.sim.pathfinder, waypoint_pose, self.meters_per_pixel)
            waypoint_pose.pop(-1)
            xy_vis_points, target_point = xy_vis_points[:-1], xy_vis_points[-1]

            # if att_features.shape[0] == len(waypoint_pose): # without global node
            att_features = att_features[-len(waypoint_pose):] # the attention scores of (i) the env global node, or (ii) goal emb or (iii) cur emb to all waypoints
            if len(waypoint_pose) > 1:
                att_features = (att_features - att_features.min()) / (att_features.max() - att_features.min())

            #print(top_down_map.shape, '\n', xy_vis_points,'\n', att_features,'\n')
            for i in range(len(waypoint_pose)):
                # cv2.putText(img=top_down_map, text='{:.3f}'.format(att_features[i].item()), org=(xy_vis_points[i][1], xy_vis_points[i][0]), \
                #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #         fontScale= min(top_down_map.shape) // 3,
                #         color=(0, 0, 0),
                #         thickness=1,
                #         lineType=cv2.LINE_AA,)
                utils.paste_overlapping_image(top_down_map, (self.square * att_features[i].item()).astype(np.uint8), (xy_vis_points[i][0], xy_vis_points[i][1]))
            
            if forget_node_indices is not None:
                half_side = self.node_side // 2 + 1
                for idx in forget_node_indices:
                    u, v = xy_vis_points[idx[1]][0], xy_vis_points[idx[1]][1]
                    cv2.rectangle(top_down_map, (v - half_side, u - half_side), (v + half_side, u + half_side), (0,0,255), 1)
                
            # elif att_features.shape[0] > len(waypoint_pose): # with global node
            #     att_features = (att_features - att_features.min()) / (att_features.max() - att_features.min())

            #     for i in range(len(waypoint_pose)):
            #         utils.paste_overlapping_image(top_down_map, (self.square * att_features[i+1].item()).astype(np.uint8), (xy_vis_points[i][0], xy_vis_points[i][1]))
                
            #     utils.paste_overlapping_image(top_down_map, (self.square * att_features[0].item()).astype(np.uint8), (0, top_down_map.shape[0] //2))

            cv2.arrowedLine(top_down_map, (xy_vis_points[-1][1], xy_vis_points[-1][0] - 20), (xy_vis_points[-1][1], xy_vis_points[-1][0]), (0,0,255), tipLength=0.3)
            utils.paste_overlapping_image(top_down_map, self.target_square  // self.num_goals * (self.curr_goal_idx+1), (target_point[0], target_point[1]))

            maps.draw_path(top_down_map, xy_vis_points, color=(112,193,179))

        #if mode == 'rgb' or mode == 'rgb_array':
            #return img
            
        if imshow:
            cv2.imshow('render', img[:,:,::-1])
            if waypoint_pose is not None:
                cv2.imshow('top_down_map', top_down_map[:,:,::-1])
            # if attn_img is not None:
            #     cv2.imshow('attn', attn_img[:,:,::-1])
            cv2.waitKey(1)
            #return img
        return img, top_down_map
        #return super().render(mode)

    def render_map_nodes(self, top_down_map, waypoint_pose, att_features, forget_node_indices):
        xy_vis_points = self.convert_points_to_topdown(self.habitat_env.sim.pathfinder, waypoint_pose, self.meters_per_pixel)
        att_features = att_features[-len(waypoint_pose):]
        if len(waypoint_pose) > 1:
                att_features = (att_features - att_features.min()) / (att_features.max() - att_features.min())

        #print(top_down_map.shape, '\n', xy_vis_points,'\n', att_features,'\n')
        maps.draw_path(top_down_map, xy_vis_points, color=(112,193,179))
        for i in range(len(waypoint_pose)):
            # cv2.putText(img=top_down_map, text='{:.3f}'.format(att_features[i].item()), org=(xy_vis_points[i][1], xy_vis_points[i][0]), \
            #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #         fontScale= min(top_down_map.shape) // 3,
            #         color=(0, 0, 0),
            #         thickness=1,
            #         lineType=cv2.LINE_AA,)
            utils.paste_overlapping_image(top_down_map, (self.square * att_features[i].item()).astype(np.uint8), (xy_vis_points[i][0], xy_vis_points[i][1]))
        
        # print(forget_node_indices)
        # print(xy_vis_points)
        if forget_node_indices is not None:
            #print('forget_node_indices',forget_node_indices)
            half_side = self.node_side // 2 + 1
            for idx in forget_node_indices:
                u, v = xy_vis_points[idx[1]][0], xy_vis_points[idx[1]][1]
                cv2.rectangle(top_down_map, (v - half_side, u - half_side), (v + half_side -1, u + half_side - 1), (255,255,255), 2)
            
        cv2.arrowedLine(top_down_map, (xy_vis_points[-1][1], xy_vis_points[-1][0] - 20), (xy_vis_points[-1][1], xy_vis_points[-1][0]), (255,255,0), tipLength=0.3)
        

    def render_simple_traj(self, render_traj=False, waypoint_pose=None, att_features=None, forget_node_indices=None):
        img = maps.get_topdown_map(self.habitat_env.sim.pathfinder, height=self.positions[0][1], meters_per_pixel=self.meters_per_pixel)
        img = self.recolor_map[img]
        
        if render_traj:
            if len(self.positions) > 2:
                traj_points = self.convert_points_to_topdown(self.habitat_env.sim.pathfinder, self.positions, self.meters_per_pixel)
                traj_points = traj_points[1:]
                for pos_id in range(1, len(traj_points)):
                    color = tuple(map(lambda x: int(255 * x), self.cmap((pos_id + 1) / len(traj_points))[0:3])) if waypoint_pose is None else (12,12,172)
                    cv2.line(img,
                            (traj_points[pos_id-1][1], traj_points[pos_id-1][0]),
                            (traj_points[pos_id][1], traj_points[pos_id][0]),
                            color=color,
                            thickness = 2)
        
        if waypoint_pose is not None:
            self.render_map_nodes(img, waypoint_pose, att_features, forget_node_indices)

        target_pos = self.convert_points_to_topdown(self.habitat_env.sim.pathfinder, [self.obs['target_pose']], self.meters_per_pixel)[0]
        utils.paste_overlapping_image(img, self.target_square  // self.num_goals * (self.curr_goal_idx+1), (target_pos[0], target_pos[1]))

        return img

    def render_detailed_traj(self, info):
        img = observations_to_image(self.obs, info, mode='panoramic')
        str_action = 'XX'
        if 'STOP' not in self.habitat_env.task.actions:
            action_list = ["MF", 'TL', 'TR']
        else:
            action_list = ["ST", "MF", 'TL', 'TR']
        if self._previous_action != -1:
            str_action = str(action_list[self._previous_action])

        reward = self.total_reward.sum() if isinstance(self.total_reward, np.ndarray) else self.total_reward
        txt = 't: %03d, r: %.2f ,dist: %.2f, stuck: %02d  a: %s '%(self.time_t,reward, self.get_dist(self.curr_goal.position)
                                                                , self.stuck, str_action)
        if self.has_log_info is not None:
            if self.has_log_info['type'] == 'str':
                txt += ' ' + self.has_log_info['info']
        elif self.return_have_been:
            txt += '                                 '
        if hasattr(self.mapper, 'node_list'):
            if self.mapper.node_list is None:
                txt += ' node : NNNN'
                txt += ' curr : NNNN'
            else:
                num_node = len(self.mapper.node_list)
                txt += ' node : %03d' % (num_node)
                curr_info = self.mapper.curr_info
                if 'curr_node' in curr_info.keys():
                    txt += ' curr: {}'.format(curr_info['curr_node'].cpu().numpy())
                if 'goal_prob' in curr_info.keys():
                    txt += ' goal %.3f'%(curr_info['goal_prob'])

        img = append_text_to_image(img, txt)

        return img
    
    def render(self, mode='rgb', waypoint_pose=None, att_features=None, forget_node_indices=None, record=0, imshow=False):
        # waypoint_pose: a list of waypoint xyz coords
        # att_features: the att scores of the last GATv2 layer; size: num_nodes
        # forget_node_indices: 
        # record: 
        # 0: no record; 1: only detailed traj rendering
        # 2: only simple traj rendering
        # 3: detailed traj rendering and goal/GATv2 att scores
        # 4: simple traj rendering and goal/GATv2 att scores
        # 5: detailed traj and simple traj rendering overlapped with goal/GATv2 att scores

        info = self.get_info(None) if self.info is None else self.info # ['distance_to_goal', 'success', 'spl', 'collisions', 'top_down_map']
        
        img = None
        if record in [1,3,5]:
            # img = cv2.resize(self.render_detailed_traj(info), dsize=(950,450))
            img = self.render_detailed_traj(info) # (310, 575, 3)

        elif record in [2,4]:
            img = self.render_simple_traj()
        
        top_down_map = None
        if record in [3,4] and att_features is not None:
            top_down_map = self.render_simple_traj(render_traj=False, waypoint_pose=waypoint_pose, att_features=att_features, forget_node_indices=forget_node_indices)
            
        elif record == 5:
            top_down_map = self.render_simple_traj(render_traj=True, waypoint_pose=waypoint_pose, att_features=att_features, forget_node_indices=forget_node_indices)
            
        if imshow:
            cv2.imshow('render', img[:,:,::-1])
            if waypoint_pose is not None:
                cv2.imshow('top_down_map', top_down_map[:,:,::-1])

            cv2.waitKey(1)

        return img, top_down_map

    def get_dists(self, pose, other_poses):
        dists = np.linalg.norm(np.array(other_poses).reshape(len(other_poses),3) - np.array(pose).reshape(1,3), axis=1)
        return dists

class MultiSearchEnv(SearchEnv):
    def step(self, action):
        if isinstance(action, dict):
            action = action['action']
        self._previous_action = action

        goal_info = self.habitat_env.task.measurements.measures['goal_index'].get_metric()

        if 'STOP' in self.action_space.spaces and action == 0:
            dist = self.get_dist(self.curr_goal.position)
            #print("action", action, "dist", dist, self.success_distance); input()
            if dist <= self.success_distance:
                #self.habitat_env.task.measurements.measures['goal_index'].increase_goal_index()
                
                all_done = self.habitat_env.task.measurements.measures['goal_index'].increase_goal_index()
                #print("update to {}/{} goal".format(self.curr_goal_idx+1, goal_info['num_goals'])); input()
                state = self.habitat_env.sim.get_agent_state()
                obs = self.habitat_env._sim.get_observations_at(state.position, state.rotation)
                obs.update(self.habitat_env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self.habitat_env.current_episode,
                    action=action,
                    task=self.habitat_env.task,
                ))
                if all_done:
                    done = True
                    reward = self.SUCCESS_REWARD
                else:
                    done = False
                    reward = 0
                # It's necessary to update info immediately; otherwise, SR and SPL will be incorrect after reaching a goal
                self.update_measures(action=self.action_dict[action])
                self.info = self.get_info(None)
                self.info["change_goal"] = True
            else:
                # if the agent calls stop before it finds its goal, then the episode will be terminated and the agent will never navigate to subsequent goals 
                obs, reward, done, self.info = super(SearchEnv, self).step(self.action_dict[action])
        else:
            obs, reward, done, self.info = super(SearchEnv, self).step(self.action_dict[action])

        self.time_t += 1
        self.info['length'] = self.time_t * done
        self.info['episode'] = int(self.current_episode.episode_id)
        self.info['distance_to_goal'] = self._previous_measure
        self.info['step'] = self.time_t
        self.info['num_goals'] = goal_info['num_goals']
        self.positions.append(self.current_position)
        self.obs = self.process_obs(obs)
        self.total_reward += reward
        self.prev_position = self.current_position.copy()
        self.prev_rotation = self.current_rotation.copy()
        if self.recording_now:
            self.imgs.append(self.render('rgb'))
            if done: self.save_video(self.imgs)
        return self.obs, reward, done, self.info
    
    # 下面这两个方法读取的是custom_habitat_task.py中GoalIndex类的成员变量
    @property
    def curr_goal_idx(self):
        if 'GOAL_INDEX' in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            goal_info = self.habitat_env.get_metrics()['goal_index']
            return min(goal_info['curr_goal_index'], goal_info['num_goals']-1)
        else:
            return 0

    def _episode_success(self):
        return self.habitat_env.task.measurements.measures['goal_index'].all_done

if __name__ == '__main__':

    from env_utils.make_env_utils import add_panoramic_camera
    from configs.default import get_config
    import numpy as np
    import os
    import time
    from habitat import make_dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    config = get_config()
    config.defrost()
    config.DIFFICULTY = 'hard'
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 10
    config.NUM_PROCESSES = 1
    config.NUM_VAL_PROCESSES = 0

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    training_scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        training_scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET) # get_scenes_to_load() is defined in line 40 in habitat/datasets/pointnav/pointnav_dataset.py
    
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = training_scenes
    config.TASK_CONFIG = add_panoramic_camera(config.TASK_CONFIG)
    config.record = True
    config.freeze()
    action_list = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

    env = SearchEnv(config)
    obs = env.reset()
    env.build_path_follower()
    img = env.render('rgb')
    done = False
    fps = {}
    reset_time = {}

    scene = env.habitat_env.current_episode.scene_id.split('/')[-2]
    fps[scene] = []
    reset_time[scene] = []
    imgs = [img]
    while True:
        best_action = env.get_best_action()
        img = env.render('rgb')
        #imgs.append(img)

        cv2.imshow('render', img[:, :, [2, 1, 0]])
        key = cv2.waitKey(0)
        #
        # if key == ord('s'): action = 1
        # elif key == ord('w'): action = 0
        # elif key == ord('a'): action = 1
        # elif key == ord('d'): action = 2
        # elif key == ord('r'):
        #     done = True
        #     print(done)
        # elif key == ord('q'):
        #     break
        # else:
        #     action = env.action_space.sample()

        if done:
            tic = time.time()
            obs = env.reset()
            toc = time.time()
            scene = env.habitat_env.current_episode.scene_id.split('/')[-2]
            fps[scene] = []
            reset_time[scene] = []
            reset_time[scene].append(toc-tic)
            done = False
            imgs = []
        else:
            tic = time.time()
            obs, reward, done, info = env.step(best_action)
            toc = time.time()
            fps[scene].append(toc-tic)
            print(toc-tic)
        #break
        if len(fps) > 20:
            break
    print('===============================')
    print('FPS : ', [(key, np.array(fps_list).mean()) for key, fps_list in fps.items()])
    print('Reset : ', [(key, np.array(reset_list).mean()) for key, reset_list in reset_time.items()])
