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

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--num', type=int, default=12)
    parser.add_argument('--result_path', type=str, default='../data/graspdata2')
    args = parser.parse_args()

    device = 'cpu'

    # hand models
    '''Robotiq Hand'''
    # hand_model = HandModel(
    #     mjcf_path='mjcf/robotiq_hand/2f85_primitified.xml',
    #     mesh_path='mjcf/robotiq_hand/meshes',
    #     contact_points_path='mjcf/robotiq_hand/contact_points.json',
    #     penetration_points_path='mjcf/robotiq_hand/penetration_points.json',
    #     n_surface_points=2000,
    #     device=device
    # )
    '''Shadow Hand'''
    # hand_model = HandModel(
    #     mjcf_path='mjcf/shadow_hand/shadow_hand_wrist_free.xml',
    #     mesh_path='mjcf/shadow_hand/meshes',
    #     contact_points_path='mjcf/shadow_hand/contact_points.json',
    #     penetration_points_path='mjcf/shadow_hand/penetration_points.json',
    #     device=device
    # )
    # joint_names = [
    #     'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    #     'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    #     'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    #     'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    #     'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    # ]
    '''Franka Hand'''
    # hand_model = HandModel(
    #     urdf_path='mjcf/franka_hand/franka.urdf',
    #     contact_points_path='mjcf/franka_hand/contact_points.json', 
    #     n_surface_points=1000, 
    #     device=device
    # )

    # joint_names = [
    #     'left_finger', 'right_finger'
    # ]
    '''Barret Hand'''
    hand_model = HandModel(
        urdf_path='mjcf/barret_hand/barret_hand_collisions_primitified.urdf',
        contact_points_path='mjcf/barret_hand/contact_points.json', 
        n_surface_points=1000, 
        device=device
    )

    joint_names = [
        'bh282_j00', 'bh282_j01', 'bh282_j02', 'bh282_j10', 'bh282_j11', 'bh282_j12', "bh282_j21", "bh282_j22", 
    ]



    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
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
