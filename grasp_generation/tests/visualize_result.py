"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go

# from utils.hand_model import HandModel
from utils.hand_model_urdf import HandModel
from utils.object_model import ObjectModel
from hands.hand_configs import *
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro_cfg')
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--num', type=int, default=5)
    args = parser.parse_args()
    hand_name = args.hand[:args.hand.rfind('_')]
    result_path = os.path.join("../output", hand_name + "_graspdata") 
    device = 'cpu'
    
    if args.hand not in ["allegro_cfg", "barrett_cfg", "franka_cfg"]:
        raise ValueError("the argument for hand is not found in hands assets")
    if args.hand == "allegro_cfg":
        hand = allegro_cfg
    elif args.hand == "barrett_cfg":
        hand = barrett_cfg
    elif args.hand == "franka_cfg":
        hand = franka_cfg
        
    hand_model = HandModel(
        urdf_path=hand["urdf_path"],
        contact_points_path=hand["contact_points_path"],
        default_pos=hand["default_pos"], 
        n_surface_points=1000, 
        device=device
    )
    
    joint_names = hand["joint_names"]


    # load results
    data_dict = np.load(os.path.join(result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    qpos = data_dict['qpos']
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + [qpos_st[name] for name in joint_names], dtype=torch.float, device=device)

    

    # object model
    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=5,
        num_samples=2000, 
        device=device
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    # visualize
    ith = 0
    if 'qpos_st' in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(i=ith, opacity=0.5, color='lightblue', with_contact_points=False)
    else:
        hand_st_plotly = []
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(i=ith, opacity=1, color='lightblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=ith, color='lightgreen', opacity=1)
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    if 'energy' in data_dict:
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_spen = round(data_dict['E_spen'], 5)
        E_joints = round(data_dict['E_joints'], 5)
        result = f'Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}'
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    fig.update_layout(scene_aspectmode='data')
    fig.show()
