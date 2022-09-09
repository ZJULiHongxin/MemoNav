#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN
import os

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.VERSION = 'base'
_C.BASE_TASK_CONFIG_PATH = "configs/vistargetnav_gibson.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.IL_TRAINER_NAME = "bc"
_C.RL_TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]

_C.TENSORBOARD_DIR = "data/logs/"
_C.VIDEO_DIR = "data/video_dir"
_C.EVAL_CKPT_PATH_DIR = "data/eval_checkpoints"  # path to ckpt or path to ckpts dir
_C.CHECKPOINT_FOLDER = "data/new_checkpoints"

_C.NUM_PROCESSES = 2
_C.NUM_VAL_PROCESSES = 0

_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]

_C.NUM_UPDATES = 10800000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.VIS_INTERVAL = 10
_C.GRAPH_TH = 0.75
_C.POLICY = 'PointNavResNetPolicy'
_C.visual_encoder_type = 'unsupervised'
_C.WRAPPER = 'EnvWrapper'
_C.BC_WRAPPER = 'BCWrapper'
_C.DIFFICULTY = 'easy'
_C.NUM_GOALS = 1
_C.NUM_AGENTS = 1
_C.scene_data = 'gibson'
_C.OBS_TO_SAVE = ['panoramic_rgb', 'panoramic_depth', 'target_goal']
_C.noisy_actuation = True
_C.USE_AUXILIARY_INFO = True

#----------------------------------------------------------------------------
# Base architecture config
_C.features = CN()
_C.features.visual_feature_dim = 512
_C.features.action_feature_dim = 32
_C.features.time_dim = 8
_C.features.hidden_size = 512
_C.features.rnn_type = 'LSTM'
_C.features.num_recurrent_layers = 2
_C.features.backbone = 'resnet18'
_C.features.message_feature_dim = 32
#----------------------------------------------------------------------------
# GCN
_C.GCN = CN()
_C.GCN.TYPE = "GCN" # "GAT", "GATv2"
_C.GCN.GRAPH_NORM = "" # "graph_norm"
_C.GCN.NUM_LAYERS = 3
_C.GCN.ENV_GLOBAL_NODE_MODE = "unavailable" # "respawn", "no_respawn", "embedding", "unavailable"
_C.GCN.RANDOMINIT_ENV_GLOBAL_NODE = False # if false, initialize the env global node as a zero vector
_C.GCN.WITH_CUROBS_GLOBAL_NODE = False
_C.GCN.ENV_GLOBAL_NODE_LINK_RANGE = -1.0 # How many nodes the global node link to. -1 means disabled. This variable is only used for ablation study. 
_C.GCN.RANDOM_REPLACE = False
# Fusion method
_C.FUSION_TYPE = "transformer" # or "two_branch", "one_branch", "transformer_wo_curobs_decoder" MLP
# Transformer
_C.transformer = CN()
_C.transformer.hidden_dim = 512
_C.transformer.dropout = 0.1
_C.transformer.nheads = 4
_C.transformer.dim_feedforward = 1024
_C.transformer.enc_layers = 2
_C.transformer.dec_layers = 1
_C.transformer.pre_norm = False
_C.transformer.num_queries = 1

_C.transformer.DECODE_GLOBAL_NODE = True # Whether or not to add the global node to the keys and values. This varibale is used for ablation

# for memory module
_C.memory = CN()
_C.memory.embedding_size = 512
_C.memory.memory_size = 100 # maximum number of nodes
_C.memory.pose_dim = 5
_C.memory.need_local_memory = False
_C.memory.FORGET = False # NOTE: innovation 2
_C.memory.FORGETTING_ATTN = "goal" # ["cur", "global_node"]
_C.memory.FORGETTING_TYPE = "simple" # ["Expire"]
_C.memory.TRAINIG_FORGET = False # use the forgetting mechanism in evaluation, not in training
# For Expire-span
_C.memory.EXPIRE_INIT_PERCENTAGE = 0.1
_C.memory.MAX_SPAN = 32
_C.memory.PRE_DIV = 6
_C.memory.RAMP = 6
_C.memory.EXPIRE_LOSS_COEF = 5e-6
# For simple
_C.memory.TOLERANCE = 10 # implement forgetting mechanism after TOLERANCE nodes have been created
_C.memory.RANK = "bottom" # or "top"
_C.memory.RANK_THRESHOLD = 0.2 # nodes whose att-scores remain in the bottom for several consecutive steps will be forgotten 
_C.memory.RANDOM_SELECT = False # Whether or not add to randomly select graph nodes. This varibale is used for ablation
# For explicit supervison on the att. score in Dec_target
_C.memory.ATTSCORE_LOSS_COEF = 0.0

_C.saving = CN()
_C.saving.name = 'test'
_C.saving.log_interval = 100
_C.saving.save_interval = 500
_C.saving.eval_interval = 500
_C.record = False
_C.render = False
_C.record_GAT_att = False

_C.RL = CN()

_C.RL.SUCCESS_MEASURE = "SUCCESS"
_C.RL.SUCCESS_DISTANCE = 1.0
_C.RL.REWARD_METHOD = 'progress'

_C.RL.SLACK_REWARD = -0.001
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.COLLISION_REWARD = -0.001

_C.RL.PPO = CN()

_C.RL.PPO.clip_param=0.2
_C.RL.PPO.ppo_epoch=2
_C.RL.PPO.num_mini_batch=128
_C.RL.PPO.value_loss_coef=0.5
_C.RL.PPO.entropy_coef=0.01
_C.RL.PPO.lr=0.00001
_C.RL.PPO.eps=0.00001
_C.RL.PPO.max_grad_norm=0.2
_C.RL.PPO.num_steps = 256
_C.RL.PPO.use_gae=True
_C.RL.PPO.gamma=0.99
_C.RL.PPO.tau=0.95
_C.RL.PPO.use_linear_clip_decay=True
_C.RL.PPO.use_linear_lr_decay=True
_C.RL.PPO.reward_window_size=50
_C.RL.PPO.use_normalized_advantage=True
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.pretrained_weights=""
_C.RL.PPO.rl_pretrained=False
_C.RL.PPO.il_pretrained=False
_C.RL.PPO.pretrained_encoder=False
_C.RL.PPO.train_encoder=True
_C.RL.PPO.reset_critic=False
_C.RL.PPO.backbone='resnet18'
_C.RL.PPO.rnn_type='LSTM'
_C.RL.PPO.num_recurrent_layers=2

_C.BC = CN()
_C.BC.lr = 0.0001
_C.BC.eps = 0.00001
_C.BC.max_grad_norm = 0.5
_C.BC.use_linear_clip_decay=True
_C.BC.use_linear_lr_decay=True
_C.BC.backbone= 'resnet18'
_C.BC.rnn_type= 'LSTM'
_C.BC.num_recurrent_layers=2
_C.BC.batch_size = 4
_C.BC.max_demo_length = 100
_C.BC.max_epoch = 100
_C.BC.lr_decay = 0.5
_C.BC.num_workers = 0

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    version = None,
    create_folders = True,
    base_task_config_path = ""
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()


    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if base_task_config_path:
        config.BASE_TASK_CONFIG_PATH = base_task_config_path
    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    #if opts:
    #    config.CMD_TRAILING_OPTS = opts
    #    config.merge_from_list(opts)
    if version is not None and version != "":
        config.VERSION = version

    if create_folders:
        if not os.path.exists('data'): os.mkdir('data')
        if not os.path.exists(config.TENSORBOARD_DIR): os.mkdir(config.TENSORBOARD_DIR)
        if not os.path.exists(config.VIDEO_DIR): os.mkdir(config.VIDEO_DIR)
        if not os.path.exists(config.EVAL_CKPT_PATH_DIR): os.mkdir(config.EVAL_CKPT_PATH_DIR)
        if not os.path.exists(config.CHECKPOINT_FOLDER): os.mkdir(config.CHECKPOINT_FOLDER)

        config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, config.VERSION)
        config.VIDEO_DIR = os.path.join(config.VIDEO_DIR, config.VERSION)
        config.EVAL_CKPT_PATH_DIR = os.path.join(config.EVAL_CKPT_PATH_DIR, config.VERSION)
        config.CHECKPOINT_FOLDER = os.path.join(config.CHECKPOINT_FOLDER, config.VERSION)
        
        if not os.path.exists(config.TENSORBOARD_DIR): os.mkdir(config.TENSORBOARD_DIR)
        if not os.path.exists(config.VIDEO_DIR): os.mkdir(config.VIDEO_DIR)
        if not os.path.exists(config.EVAL_CKPT_PATH_DIR): os.mkdir(config.EVAL_CKPT_PATH_DIR)
        if not os.path.exists(config.CHECKPOINT_FOLDER): os.mkdir(config.CHECKPOINT_FOLDER)

    config.freeze()
    return config
