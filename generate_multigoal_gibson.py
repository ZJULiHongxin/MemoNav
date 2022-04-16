import os, random, gzip, shutil, json
import habitat, habitat_sim
from habitat.utils.visualizations import maps, utils
from habitat_sim.utils import common
import time
import imageio
import gzip
import quaternion as q
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import magnum as mn
import cv2
import platform
if "windows" not in platform.platform().lower():
    matplotlib.use('Agg') 

os.environ['GLOG_minloglevel'] = "3"
os.environ['HABITAT_SIM_LOG'] = "quiet"
os.environ['MAGNUM_LOG'] = "quiet"

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)

def calc_geodesic_dist(path, pathfinder, p1: list, p2: list):
    path.requested_start = p1
    path.requested_end = p2
    pathfinder.find_path(path)
    return path.geodesic_distance

if __name__ == "__main__":
    scene_dir = "/data/jing_li/habitat-lab/data/scene_datasets/gibson_habitat"
    scene_name_list = ['Edgemere', 'Eastville', 'Greigsville', 'Swormville', 'Sands', 'Ribera', 'Scioto', 'Pablo', 'Elmira', 'Mosquito', 'Denmark', 'Sisters', 'Cantwell', 'Eudora']
    minmax_dist = {
        "Edgemere": 7.17, "Eastville": 14.84, "Greigsville": 8.40, "Swormville": 11.17, "Sands": 9.55, "Ribera": 9.72, "Scioto": 13.45, "Pablo": 9.46,
        "Elmira": 8.51, "Mosquito": 24.29, "Denmark": 8.12, "Sisters": 12.09, "Cantwell": 15.92, "Eudora": 7.51
    }
    saved_dir = './image-goal-nav-dataset/val_multigoal'
    if os.path.exists(saved_dir): shutil.rmtree(saved_dir)
    os.mkdir(saved_dir)

    info_dir = './image-goal-nav-dataset/val_multigoal_info'
    if os.path.exists(info_dir): shutil.rmtree(info_dir)
    os.mkdir(info_dir)
    dataset_info_txt = open(os.path.join(info_dir, "info.txt"), 'w')
    
    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}

    meters_per_pixel = 0.01  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

    max_search_radius = 1.0
    num_episodes = 50
    save_img = 1
    num_goals = 3
    num_sampling = 100
    tolerance = 600
    dist_upper, obstacle_dist, proximity = 10, 0.5, 2
    start_square = np.tile(np.array([[[255,0,0]]], dtype=np.uint8), (12, 12, 1))
    target_square = np.tile(np.array([[[0,255,0]]], dtype=np.uint8), (12, 12, 1))

    path = habitat_sim.ShortestPath()
    geodesic_dist_lst = []

    t = time.time()

    for i in range(len(scene_name_list)):
        
        scene_name = scene_name_list[i]
        #if scene_name != "Ribera": continue
        scene_path = os.path.join(scene_dir, scene_name + ".glb")
        
        if save_img == 1:
            save_img_dir = os.path.join(info_dir, scene_name)
            if os.path.exists(save_img_dir): shutil.rmtree(save_img_dir)
            os.mkdir(save_img_dir)
        
        sim_settings = {
            "width": 256,  # Spatial resolution of the observations
            "height": 256,
            "scene": scene_path,  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "color_sensor": rgb_sensor,  # RGB sensor
            "depth_sensor": depth_sensor,  # Depth sensor
            "semantic_sensor": semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
        }

        cfg = make_cfg(sim_settings)
        sim = habitat_sim.Simulator(cfg)

        print("[{}/{}] Generating {} trajectories. Loading scene mesh from {}".format(i, len(scene_name_list), scene_name, scene_path))

        random.seed(sim_settings["seed"])
        sim.seed(sim_settings["seed"])
        sim.pathfinder.seed(sim_settings["seed"])

        with gzip.open(os.path.join(saved_dir, scene_name+".json.gz"), 'w') as f:
            ep_list = []
            for j in range(num_episodes):    
                while True:
                    goal_list = []
                    geodesic_dist_list = []
                    # select a starting position
                    while True:
                        starting_point = sim.pathfinder.get_random_navigable_point()
                        dist2obstacle = sim.pathfinder.distance_to_closest_obstacle(starting_point, max_search_radius)
                        if dist2obstacle > obstacle_dist: break
                    
                    goal_list.append(starting_point)
                    
                    success = False
                    for goal_idx in range(num_goals):
                        max_dist = 0
                        confirmed_next_goal = None

                        if goal_idx != num_goals - 1:
                            cnt, n = 0 , 0
                            while cnt < num_sampling and n <= tolerance:
                                n += 1
                                next_goal = sim.pathfinder.get_random_navigable_point()
                                geodesic_dist = calc_geodesic_dist(path, sim.pathfinder, next_goal, goal_list[-1])
                                # print("\n==================================")
                                # print(sim.pathfinder.distance_to_closest_obstacle(next_goal, max_search_radius), obstacle_dist)
                                # print(geodesic_dist, dist_upper)
                                # print(geodesic_dist == float("inf"))
                                # print(next_goal[1], goal_list[-1][1])
                                # if scene_name == 'Sands' and (n%50 ==0 or n>550): 
                                #     print(n,
                                # sim.pathfinder.distance_to_closest_obstacle(next_goal, max_search_radius), obstacle_dist,
                                # geodesic_dist, dist_upper,
                                # next_goal[1], goal_list[-1][1])
                                if sim.pathfinder.distance_to_closest_obstacle(next_goal, max_search_radius) < obstacle_dist \
                                    or geodesic_dist == float("inf") \
                                    or geodesic_dist > dist_upper \
                                    or next_goal[1] != goal_list[-1][1]: continue

                                dist_avg = 0
                                for goal in goal_list:
                                    dist_avg += calc_geodesic_dist(path, sim.pathfinder, goal, next_goal)
                                
                                dist_avg /= len(goal_list)
                                if dist_upper > dist_avg > max_dist:
                                    confirmed_next_goal = next_goal
                                    max_dist = dist_avg

                                cnt += 1
                            
                        else: # The final goal must be in the vicinity of previous goals or the starting point
                            n = 0
                            while n <= tolerance:
                                n += 1

                                next_goal = sim.pathfinder.get_random_navigable_point()
                                geodesic_dist = calc_geodesic_dist(path, sim.pathfinder, next_goal, goal_list[-1])
                                if sim.pathfinder.distance_to_closest_obstacle(next_goal, max_search_radius) < obstacle_dist \
                                    or geodesic_dist == float("inf") \
                                    or geodesic_dist > dist_upper \
                                    or next_goal[1] != goal_list[-1][1]: continue

                                for goal in goal_list[:-1]:
                                    if calc_geodesic_dist(path, sim.pathfinder, goal, next_goal) < proximity:
                                        confirmed_next_goal = next_goal
                                        break
                                
                                if confirmed_next_goal is not None: break
                        
                        if confirmed_next_goal is None: break

                        goal_list.append(confirmed_next_goal)
                        geodesic_dist_list.append(calc_geodesic_dist(path, sim.pathfinder, goal_list[-1], goal_list[-2]))
                        # print("found_path : " + str(found_path))
                        # print("geodesic_distance : " + str(geodesic_distance))
                        # print("path_points : " + str(path.points))
                    else:
                        success = True
                    
                    if success:
                        break
                    else:
                        print("Cannot find a goal. Reselect the starting point.")
                    
                    
                tangent_orientation_matrix = mn.Matrix4.look_at(
                        goal_list[0], goal_list[1], np.array([0, 1.0, 0])
                    )
                tangent_orientation_q = mn.Quaternion.from_matrix(
                    tangent_orientation_matrix.rotation()
                )
                rotation = q.as_float_array(common.quat_from_magnum(tangent_orientation_q))

                geodesitc_dist_sum = sum(geodesic_dist_list)
                geodesic_dist_lst.append(geodesitc_dist_sum)

                ep = {
                    "episode_id": j,
                    "scene_id": "data/scene_datasets/gibson/{}.glb".format(scene_name),
                    "start_position": goal_list[0].tolist(),
                    "start_rotation": rotation.tolist()[::-1], # it is weird to have to reverse this quarternion
                    "info": {
                        "geodesic_distance_sum": geodesitc_dist_sum,
                        "geodesic_distance": geodesic_dist_list,
                        "difficulty": "multigoal"
                        },
                    "goals": [ {"position": x.tolist(), "radius": None} for x in goal_list[1:]],
                    "shortest_paths": None,
                    "start_room": None
                }
                
                ep_list.append(ep)
                if save_img:
                    xy_vis_points = convert_points_to_topdown(
                        sim.pathfinder, goal_list, meters_per_pixel
                    )
                    # use the y coordinate of the sampled nav_point for the map height slice
                    top_down_map = maps.get_topdown_map(
                        sim.pathfinder, height=goal_list[1][1], meters_per_pixel=meters_per_pixel
                    )
                    recolor_map = np.array(
                        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                    )
                    top_down_map = recolor_map[top_down_map]
                    # input(xy_vis_points)
                    utils.paste_overlapping_image(top_down_map, start_square, (int(xy_vis_points[0][1]), int(xy_vis_points[0][0])))
                    for goal_idx in range(1,len(xy_vis_points)):
                        utils.paste_overlapping_image(top_down_map, target_square // num_goals * goal_idx, (int(xy_vis_points[goal_idx][1]), int(xy_vis_points[goal_idx][0])))
                        cv2.putText(img=top_down_map, text='{}'.format(goal_idx), org=(int(xy_vis_points[goal_idx][0]), int(xy_vis_points[goal_idx][1])), \
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= min(top_down_map.shape) // 3,
                            color=(0, 0, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA,)
                    imageio.imsave(os.path.join(save_img_dir, 'ep{}.png'.format(j)), top_down_map)

                #if (j+1) % 10 == 0: print("{} trajectorys done".format(j+1))
            
            data = json.dumps(ep_list, indent=2) + '\n'
            f.write(data.encode('utf-8'))
        #if i == 1:break
        s = "[{}/{}] {} Avg total geodesic distance from start to final goal: {:.2f}\n".format(i, len(scene_name_list), scene_name, sum(geodesic_dist_lst[-num_episodes:]) / num_episodes)
        print(s)
        dataset_info_txt.write(s)
        sim.close() # this line is indispensible. 
    
    avg_total_geodesic_dist = sum(geodesic_dist_lst) / len(geodesic_dist_lst)
    s = "{} scenes, {} trajectories in total. Avg total geodesic distance from start to final goal: {:.2f}. Time elapsed: {:.2f}s\n".format(len(scene_name_list), len(geodesic_dist_lst), avg_total_geodesic_dist, time.time()-t)
    print(s)
    dataset_info_txt.write(s)
    dataset_info_txt.close()

    num_bins = 15
    geodesic_dist_range = list(range(0,2 * num_bins,2))
    geodesic_dist_histogram = [0 for _ in range(len(geodesic_dist_range)-1)]

    for dist in geodesic_dist_lst:
        for bin_id in range(len(geodesic_dist_histogram)):
            if geodesic_dist_range[bin_id] < dist <= geodesic_dist_range[bin_id+1]:
                geodesic_dist_histogram[bin_id] +=1
                break
    
    with open(os.path.join(info_dir, "geodesic_dist_lst.txt"), 'w') as f:
        for d in geodesic_dist_lst:
            f.write("{:.4f}\n".format(d))
    
    plt.figure(dpi=150)
    plt.subplots_adjust(top=0.88, bottom=0.188, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
    X = geodesic_dist_range[:-1]
    Y = geodesic_dist_histogram
    plt.bar(X, Y, width = 1.5)
    legend_lst = ["{}-{}".format(geodesic_dist_range[i], geodesic_dist_range[i+1]) for i in range(len(geodesic_dist_histogram))]
    plt.xticks(X, legend_lst, rotation=270)
    plt.xlabel('Range of Geodesic distance')
    plt.ylabel('Number of trajectories')
    plt.title('Histogram of geodesic distance over all scenes\nAvg geodesic distance: {:.2f}m'.format(avg_total_geodesic_dist))

    for x,y in zip(X,Y):
        plt.text(x+0.005,y+0.005,str(y), fontsize=6, ha='center',va='bottom')
    
    plt.savefig(os.path.join(info_dir,'geodesic_dist_histogram.jpg'))
    plt.show()