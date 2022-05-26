from gym.wrappers.monitor import Wrapper
from gym.spaces.box import Box
import torch
import numpy as np
from utils.ob_utils import log_time
TIME_DEBUG = False
from utils.ob_utils import batch_obs
import torch.nn as nn
import torch.nn.functional as F
from model.PCL.resnet_pcl import resnet18
import os
# this wrapper comes after vectorenv
from habitat.core.vector_env import VectorEnv
from env_utils.env_wrapper.graph import Graph

# To learn the functionalities of Wrapper class, see https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/
class GraphWrapper(Wrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,envs, exp_config):
        self.exp_config = exp_config
        self.envs = envs # SearchEnv or MultiSearchEnv inherited from RLEnv inherited from gym.Env
        self.env = self.envs
        if isinstance(envs,VectorEnv):
            self.is_vector_env = True
            self.num_envs = self.envs.num_envs
            self.action_spaces = self.envs.action_spaces
            self.observation_spaces = self.envs.observation_spaces
        else:
            self.is_vector_env = False
            self.num_envs = 1

        self.B = self.num_envs
        self.scene_data = exp_config.scene_data
        self.input_shape = (64, 256)
        self.feature_dim = 512
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'

        self.scene_data = exp_config.scene_data

        self.visual_encoder_type = 'unsupervised'
        self.visual_encoder = self.load_visual_encoder(self.visual_encoder_type, self.input_shape, self.feature_dim).to(self.torch_device)
        self.th = getattr(exp_config, 'graph_th', 0.75)
        self.graph = Graph(exp_config, self.B, self.torch_device)
        self.need_goal_embedding = 'wo_Fvis' in exp_config.POLICY
 
        if isinstance(envs, VectorEnv):
            for obs_space in self.observation_spaces:
                obs_space.spaces.update(
                    {'global_memory': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.feature_dim),
                                          dtype=np.float32),
                     'global_mask': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32),
                     'global_A': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.graph.M), dtype=np.float32),
                     'global_time': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32)
                     }
                )
                if self.need_goal_embedding:
                    obs_space.spaces.update(
                        {'goal_embedding': Box(low=-np.Inf, high=np.Inf, shape=(self.feature_dim,), dtype=np.float32)}
                    )                     
        self.num_agents = exp_config.NUM_AGENTS
        
        self.localize_mode = 'predict'
        
        # for forgetting mechanism
        self.forget = self.exp_config.memory.FORGET and self.exp_config.memory.FORGETTING_TYPE == "simple"
        self.forgetting_recorder = None
        self.forget_node_indices = None

        self.reset_all_memory()
    
    def load_visual_encoder(self, type, input_shape, feature_dim):
        visual_encoder = resnet18(num_classes=feature_dim)
        dim_mlp = visual_encoder.fc.weight.shape[1]
        visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
        ckpt_pth = os.path.join('model/PCL', 'PCL_encoder.pth')
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        visual_encoder.load_state_dict(ckpt)
        visual_encoder.eval()
        return visual_encoder

    def reset_all_memory(self, B=None):
        self.graph.reset(B)

        if self.forget:
            self.start_to_forget = self.exp_config.memory.TOLERANCE
            self.rank_type = self.exp_config.memory.RANK
            self.forgetting_recorder = torch.zeros(self.B, self.graph.M, self.start_to_forget, dtype=bool, device=self.torch_device)
            self.forget_node_indices = torch.ones(self.B, self.graph.M, device=self.torch_device)

            self.cur = 0
            self.forget_th = self.exp_config.memory.RANK_THRESHOLD
            
    def is_close(self, embed_a, embed_b, return_prob=False):
        with torch.no_grad():
            logits = torch.matmul(embed_a.unsqueeze(1), embed_b.unsqueeze(2)).squeeze(2).squeeze(1)
            close = (logits > self.th).detach().cpu()
        if return_prob: return close, logits
        else: return close

    # assume memory index == node index
    def localize(self, new_embedding, position, time, done_list):
        # The position is only used for visualizations.
        # done_list contains all Trues when navigation starts

        done = np.where(done_list)[0] # 一个参数np.where(arry)：输出arry中‘真’值的坐标(‘真’也可以理解为非零)

        if len(done) > 0:
            for b in done:
                self.graph.reset_at(b)
                self.graph.initialize_graph(b, new_embedding, position)

        close = self.is_close(self.graph.last_localized_node_embedding, new_embedding, return_prob=False)
        found = torch.tensor(done_list) + close # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        found_batch_indices = torch.where(found)[0]

        localized_node_indices = torch.ones([self.B], dtype=torch.int32) * -1
        localized_node_indices[found_batch_indices] = self.graph.last_localized_node_idx[found_batch_indices]
        
        # 图更新条件一：如果当前时刻智能体和上一时刻位置相同，则更新所处结点的视觉特征
        # Only time infos are updated as no embeddings are provided
        self.graph.update_nodes(found_batch_indices, localized_node_indices[found_batch_indices], time[found_batch_indices])
        
        # 以下是图更新条件二和三
        # first prepare all available nodes as 0s, and secondly set visited nodes as 1s
        # graph_mask中将每个导航进程的所有现存地图结点都用1来表示
        check_list = 1 - self.graph.graph_mask[:, :self.graph.num_node_max()]
        check_list[range(self.B), self.graph.last_localized_node_idx.long()] = 1.0

        check_list[found_batch_indices] = 1.0
        check_list
        to_add = torch.zeros(self.B)
        hop = 1
        max_hop = 0
        while not found.all():
            if hop <= max_hop : k_hop_A = self.graph.calculate_multihop(hop)
            not_found_batch_indicies = torch.where(~found)[0]
            neighbor_embedding = []
            batch_new_embedding = []
            num_neighbors = []
            neighbor_indices = []
            for b in not_found_batch_indicies:
                if hop <= max_hop:
                    neighbor_mask = k_hop_A[b,self.graph.last_localized_node_idx[b]] == 1
                    not_checked_yet = torch.where((1 - check_list[b]) * neighbor_mask[:len(check_list[b])])[0]
                else:
                    not_checked_yet = torch.where((1-check_list[b]))[0]
                neighbor_indices.append(not_checked_yet)
                neighbor_embedding.append(self.graph.graph_memory[b, not_checked_yet])
                num_neighbors.append(len(not_checked_yet))
                if len(not_checked_yet) > 0:
                    batch_new_embedding.append(new_embedding[b:b+1].repeat(len(not_checked_yet),1))
                else:
                    found[b] = True
                    to_add[b] = True
            if torch.sum(torch.tensor(num_neighbors)) > 0:
                neighbor_embedding = torch.cat(neighbor_embedding)
                batch_new_embedding = torch.cat(batch_new_embedding)
                batch_close, batch_prob = self.is_close(neighbor_embedding, batch_new_embedding, return_prob=True)
                close = batch_close.split(num_neighbors)
                prob = batch_prob.split(num_neighbors)

                for ii in range(len(close)):
                    is_close = torch.where(close[ii] == True)[0]
                    if len(is_close) == 1:
                        found_node = neighbor_indices[ii][is_close.item()]
                    elif len(is_close) > 1:
                        found_node = neighbor_indices[ii][prob[ii].argmax().item()]
                    else:
                        found_node = None
                    b = not_found_batch_indicies[ii]
                    if found_node is not None:
                        found[b] = True
                        localized_node_indices[b] = found_node

                        # 图更新条件二： If the current location and the last localized node are different, a new edge between vi and vn is added.
                        # The embedding of vi is replaced with the current feature
                        if found_node != self.graph.last_localized_node_idx[b]:
                            self.graph.update_node(b, found_node, time[b], new_embedding[b])
                            self.graph.add_edge(b, found_node, self.graph.last_localized_node_idx[b])
                            self.graph.record_localized_state(b, found_node, new_embedding[b])

                            if self.forget == True:
                                self.forget_node_indices[b,found_node] = 1
                                #self.forget_node_indices.discard((b, found_node))
                                self.forgetting_recorder[b,found_node] = False
                            
                    check_list[b, neighbor_indices[ii]] = 1.0
            hop += 1

        # 图更新条件三：If the current location cannot be localized in the VGM, a new node vNt+1 with embedding et and an edge between the new node and vn are added to the VGM.
        batch_indices_to_add_new_node = torch.where(to_add)[0]

        for b in batch_indices_to_add_new_node:
            new_node_idx = self.graph.num_node(b) # 图结点从0开始编号
            self.graph.add_node(b, new_node_idx, new_embedding[b], time[b], position[b])
            self.graph.add_edge(b, new_node_idx, self.graph.last_localized_node_idx[b])
            self.graph.record_localized_state(b, new_node_idx, new_embedding[b])

    def update_graph(self):
        if self.is_vector_env:
            args_list = [{'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b], 'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]},
                          } for b in range(self.B)]
            self.envs.call(['update_graph']*self.B, args_list)
        else:
            b = 0
            input_args = {'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b],'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]}}
            self.envs.update_graph(**input_args)

    def embed_obs(self, obs_batch):
        with torch.no_grad():
            img_tensor = torch.cat((obs_batch['panoramic_rgb']/255.0, obs_batch['panoramic_depth']),3).permute(0,3,1,2)
            vis_embedding = nn.functional.normalize(self.visual_encoder(img_tensor).view(self.B,-1),dim=1)
        return vis_embedding.detach()

    def embed_target(self, obs_batch):
        with torch.no_grad():
            img_tensor = obs_batch['target_goal'].permute(0,3,1,2)
            vis_embedding = nn.functional.normalize(self.visual_encoder(img_tensor).view(self.B,-1),dim=1)
        return vis_embedding.detach()

    def update_obs(self, obs_batch, global_memory_dict):
        # add memory to obs
        obs_batch.update(global_memory_dict)
        obs_batch.update({'localized_idx': self.graph.last_localized_node_idx.unsqueeze(1)})
        if 'distance' in obs_batch.keys():
            obs_batch['distance'] = obs_batch['distance']#.unsqueeze(1)
        if self.need_goal_embedding:
            obs_batch['goal_embedding'] = self.embed_target(obs_batch)
        return obs_batch

    def step(self, actions):

        if self.is_vector_env:
            dict_actions = [{'action': actions[b]} for b in range(self.B)]
            outputs = self.envs.step(dict_actions)
        else:
            outputs = [self.envs.step(actions)]

        obs_list, reward_list, done_list, info_list = [list(x) for x in zip(*outputs)]
        obs_batch = batch_obs(obs_list, device=self.torch_device)

        curr_vis_embedding = self.embed_obs(obs_batch)
        self.localize(curr_vis_embedding, obs_batch['position'].detach().cpu().numpy(), obs_batch['step'], done_list)

        global_memory_dict = self.get_global_memory()
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        self.update_graph()

        if self.forget:
            for b in range(self.B):
                if info_list[b]['change_goal']:
                    #input("======remember all=======")
                    self.forget_node_indices[b] = 1
                    self.forgetting_recorder[b] = False


        if self.is_vector_env:
            return obs_batch, reward_list, done_list, info_list
        else:
            return obs_batch, reward_list[0], done_list[0], info_list[0]

    def reset(self):
        obs_list = self.envs.reset()
        if not self.is_vector_env: obs_list = [obs_list]
        obs_batch = batch_obs(obs_list, device=self.torch_device)
        curr_vis_embeddings = self.embed_obs(obs_batch)
        if self.need_goal_embedding: obs_batch['curr_embedding'] = curr_vis_embeddings
        # posiitons are obtained by calling habitat_env.sim.get_agent_state().position
        self.localize(curr_vis_embeddings, obs_batch['position'].detach().cpu().numpy(), obs_batch['step'], [True]*self.B)

        if self.forget:
            self.forgetting_recorder = torch.zeros(self.B, self.graph.M, self.start_to_forget, dtype=bool, device=self.torch_device)
            #self.forget_node_indices = set()
            self.forget_node_indices = torch.ones(self.B, self.graph.M, device=self.torch_device)
        global_memory_dict = self.get_global_memory()

        # obs_batch contains following keys:
        # ['rgb_0'~'rgb_11', 'depth_0'~'depth_11', 'panoramic_rgb', 'panoramic_depth',
        # 'target_goal', 'episode_id', 'step', 'position', 'rotation', 'target_pose', 'distance', 'have_been',
        # 'target_dist_score', 'global_memory', 'global_act_memory', 'global_mask', 'global_A', 'global_time', 'forget_mask', 'localized_idx']
        # NOTE: if multiple goals are set, target_goal will have a shape [B, num_goals, 64, 252, 4]
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        
        self.update_graph()

        return obs_batch

    def forget_node(self, att_scores, num_nodes, att_type):
        if 'gat' in att_type.lower():
            return self.forget_node_gat(att_scores, num_nodes)
        else:
            return self.forget_node_transformer(att_scores, num_nodes)
    
    def forget_node_gat(self, att_scores, num_nodes):
        # att_scores: B x max_num_nodes or list of att score vectors
        
        # num_nodes: B
        if not self.forget: return None

        self.forgetting_recorder[:] = False
        self.forget_node_indices[:] = 1
        # att_scores: B x 1 x num_nodes
        #print("\n",att_scores)
        # if self.forget_th is in (0,1), then it means the proportion of nodes that should be forgotten or kept
        # if self.forget_th > 1, then it means how many nodes should be forgotten or kept
        for b in range(len(att_scores)):
            num_node = num_nodes[b].int()
            if self.rank_type=="bottom":
                keep_num = int(self.forget_th * num_node.item()) if self.forget_th < 1 else int(self.forget_th)
                forget_range = torch.arange(0, keep_num)
            elif self.rank_type=="top":
                keep_num = int(self.forget_th * num_node.item()) if self.forget_th < 1 else int(self.forget_th) # + 1 means rounding up
                forget_range = torch.arange(0, max(num_node - keep_num, 0))

            #print(forget_range, keep_num)
            forget_ids = torch.argsort(att_scores[b][-num_node:])[forget_range] # NOTE: att_scores may contain the score for the global node, and this score should be excluded

            if len(forget_ids) == 0: continue
            self.forgetting_recorder[b, forget_ids, self.cur] = True
        
        self.cur = (self.cur + 1) % self.start_to_forget

        forget_node_indices = torch.nonzero(torch.all(self.forgetting_recorder, dim=2)) # number of nodes to be forgotten

        # print(self.forgetting_recorder[0:2,0:10,:])
        # input(forget_node_indices)
        for idx in forget_node_indices:
            self.forget_node_indices[idx[0], idx[1]] = 0 # 0 means this node will be forgotten
        
        return forget_node_indices

    def forget_node_transformer(self, att_scores, num_nodes):
        # NOTE: This method only supports single-batch input
        # att_scores: B(1) x max_num_nodes      it may contain the score of the global node
        # num_nodes: B(1)
        if not self.forget: return
        #print('\n',att_scores)
        B = att_scores.shape[0]
        for b in range(B):
            num_node = num_nodes[b].int()
            att_scores_b = att_scores[b, -num_node:]

        #print(att_scores)
        # att_scores: B x 1 x num_nodes

        # if self.forget_th is in (0,1), then it means the proportion of nodes that should be forgotten or kept
        # if self.forget_th > 1, then it means how many nodes should be forgotten or kept
        # num_nodes = att_scores.shape[-1]
            if self.rank_type=="bottom":
                keep_num = int(self.forget_th * num_node) if self.forget_th < 1 else int(self.forget_th)
                forget_range = torch.arange(0, keep_num)
            elif self.rank_type=="top":
                keep_num = int(self.forget_th * num_node) + 1 if self.forget_th < 1 else int(self.forget_th) # + 1 means rounding up
                forget_range = torch.arange(0, max(num_node - keep_num, 0))

            # print('keep_num', keep_num)
            # print('forget_range',att_scores_b.shape, forget_range)
            forget_ids = torch.argsort(att_scores_b, dim=0)[forget_range]
            # print('forget_ids',forget_ids)

            if len(forget_ids) == 0: continue
            self.forgetting_recorder[b, forget_ids, self.cur] = True
            self.cur = (self.cur + 1) % self.start_to_forget

        forget_node_indices = torch.nonzero(torch.all(self.forgetting_recorder, dim=2)) # number of nodes to be forgotten

        # print("\n",att_scores[0:2,0,0:10])
        # print(self.forgetting_recorder[0:2,0:10,:])
        # input(forget_node_indices[0:2])
        for idx in forget_node_indices:
            self.forget_node_indices[idx[0], idx[1]] = 0 # 0 means this node will be forgotten
        #     self.forget_node_indices.add(tuple(forget_node_indices[i].tolist()))
        
        # 几种遗忘结点的方案：① 注意力分数绝对值低于0.2；② 注意力分数排名连续几次排在后20%；③ 连续几次；④ 累计几次
        # 连续几次排名靠后
        # self.forgetting_recorder[:,:,self.cur] = False # 清除记录，以保证连续几次不达标才会遗忘；不清除，则变为累计几次
        return forget_node_indices
    
    def forget_node_transformer1(self, att_scores, num_nodes):
        # NOTE: This method only supports single-batch input
        # att_scores: B(1) x max_num_nodes      it may contain the score of the global node
        # num_nodes: B(1)
        if not self.forget: return
        
        num_nodes = num_nodes.squeeze().int()
        #print('\n',att_scores)
        att_scores = att_scores.squeeze(0)[-num_nodes:]
        #print(att_scores)
        # att_scores: B x 1 x num_nodes

        # if self.forget_th is in (0,1), then it means the proportion of nodes that should be forgotten or kept
        # if self.forget_th > 1, then it means how many nodes should be forgotten or kept
        # num_nodes = att_scores.shape[-1]

        if self.rank_type=="bottom":
            keep_num = int(self.forget_th * num_nodes) if self.forget_th < 1 else int(self.forget_th)
            forget_range = torch.arange(-num_nodes, -num_nodes+keep_num)
        elif self.rank_type=="top":
            keep_num = int(self.forget_th * num_nodes) + 1 if self.forget_th < 1 else int(self.forget_th) # + 1 means rounding up
            forget_range = torch.arange(-num_nodes, -num_nodes+max(num_nodes - keep_num, 0))
        #print('num_nodes',num_nodes)
        forget_ids = torch.argsort(att_scores, dim=1)[:,forget_range]
        #print('forget_ids',forget_ids)
        for b in range(len(att_scores)):
            if len(forget_ids[b]) == 0: continue
            self.forgetting_recorder[b, forget_ids[b], self.cur] = True
        self.cur = (self.cur + 1) % self.start_to_forget

        forget_node_indices = torch.nonzero(torch.all(self.forgetting_recorder, dim=2)) # number of nodes to be forgotten

        # print("\n",att_scores[0:2,0,0:10])
        # print(self.forgetting_recorder[0:2,0:10,:])
        # input(forget_node_indices[0:2])
        for idx in forget_node_indices:
            self.forget_node_indices[idx[0], idx[1]] = 0 # 0 means this node will be forgotten
        #     self.forget_node_indices.add(tuple(forget_node_indices[i].tolist()))
        
        # 几种遗忘结点的方案：① 注意力分数绝对值低于0.2；② 注意力分数排名连续几次排在后20%；③ 连续几次；④ 累计几次
        # 连续几次排名靠后
        # self.forgetting_recorder[:,:,self.cur] = False # 清除记录，以保证连续几次不达标才会遗忘；不清除，则变为累计几次
        return forget_node_indices

    def get_global_memory(self, mode='feature'):
        self.graph
        global_memory_dict = {
            'global_memory': self.graph.graph_memory,
            'global_act_memory': self.graph.graph_act_memory,
            'global_mask': self.graph.graph_mask,
            'global_A': self.graph.A,
            'global_time': self.graph.graph_time,
            'forget_mask': self.forget_node_indices
        }
        return global_memory_dict

    def call(self, aa, bb):
        return self.envs.call(aa,bb)
    def log_info(self,log_type='str', info=None):
        return self.envs.log_info(log_type, info)

    @property
    def habitat_env(self): return self.envs.habitat_env
    @property
    def noise(self): return self.envs.noise
    @property
    def current_episode(self):
        if self.is_vector_env: return self.envs.current_episodes
        else: return self.envs.current_episode
    @property
    def current_episodes(self):
        return self.envs.current_episodes



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    from configs.default import get_config
    from env_utils.task_search_env import SearchEnv
    from env_utils.make_env_utils import construct_envs, make_env_fn
    config = get_config()
    config.defrost()
    config.NUM_PROCESSES = 3
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP","MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.render = True
    config.render_map = True
    config.DIFFICULTY = 'hard'
    config.WRAPPER = 'GraphWrapper'
    if torch.cuda.device_count() <= 1:
        config.TORCH_GPU_ID = 0
        config.SIMULATOR_GPU_ID = 0
    config.freeze()

    env = construct_envs(config, SearchEnv, make_env_fn=make_env_fn)
    obs = env.reset()
    env.envs.call(["build_path_follower"]*env.B)
    done = False
    imgs = []
    vid_num = 0
    stuck = 0
    import time

    total_time_dict = {}
    iter = 0
    while True:
        acts = env.envs.call(['get_best_action']*env.B)
        actions = []
        for a in acts:
             if a is not None:
                 actions.append(a)
             else:
                 actions.append(0)

        tic = time.time()
        obs, reward, done, info = env.step(actions)
        toc = time.time()

        env.envs.render('human')

        # if done[0]:
        #     video_name = 'graph_wrapper_video_%d.mp4'%vid_num
        #     with imageio.get_writer(video_name, fps=30) as writer:
        #         im_shape = imgs[-1].shape
        #         for im in imgs:
        #             if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
        #                 im = cv2.resize(im, (im_shape[1], im_shape[0]))
        #             writer.append_data(im.astype(np.uint8))
        #     writer.close()
        #     vid_num += 1
        #     imgs = []
        iter += 1

        if vid_num == 100: break

