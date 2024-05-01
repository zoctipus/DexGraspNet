"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: visualize hand model
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

import numpy as np
import torch
import argparse
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
from utils.hand_model_urdf import HandModel
from hands.hand_configs import *


torch.manual_seed(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='barrett_cfg')
    args = parser.parse_args()
    device = torch.device('cpu')
    
    if args.hand not in ["allegro_cfg", "barrett_cfg", "franka_cfg"]:
        raise ValueError("the argument for hand is not found in hands assets")
    if args.hand == "allegro_cfg":
        hand = allegro_cfg
    elif args.hand == "barrett_cfg":
        hand = barrett_cfg
    elif args.hand == "franka_cfg":
        hand = franka_cfg
    
    rot = transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes='rzyz')
    
    hand_model = HandModel(
        urdf_path=hand["urdf_path"],
        contact_points_path=hand["contact_points_path"],
        default_pos=hand["default_pos"],
        n_surface_points=1000, 
        device=device
    )
    
    hand_pose = torch.cat([
        torch.tensor([0, 0, 0], dtype=torch.float, device=device), 
        # torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=device),
        torch.tensor(rot.T.ravel()[:6], dtype=torch.float, device=device),
        # torch.zeros([16], dtype=torch.float, device=device),
        torch.tensor(hand["default_pos"], dtype=torch.float, device=device), ], dim=0)
    
    
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # info
    contact_candidates = hand_model.get_contact_candidates()[0]
    surface_points = hand_model.get_surface_points()[0]
    print(f'n_dofs: {hand_model.n_dofs}')
    print(f'n_contact_candidates: {len(contact_candidates)}')
    print(f'n_surface_points: {len(surface_points)}')
    print(hand_model.chain.get_joint_parameter_names())

    # visualize

    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue')
    v = contact_candidates.detach().cpu()
    contact_candidates_plotly = [go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='white'))]
    v = surface_points.detach().cpu()
    surface_points_plotly = [go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='lightblue'))]
    
    fig = go.Figure(hand_plotly + contact_candidates_plotly + surface_points_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()