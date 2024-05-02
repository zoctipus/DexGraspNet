
allegro_cfg={
    "urdf_path" : 'hands/allegro_hand/allegro_hand_description_left.urdf',
    "contact_points_path" : 'hands/allegro_hand/contact_points.json',
    "default_pos" : [0, 0.5, 0, 0, 
                    0, 0.5, 0, 0, 
                    0, 0.5, 0, 0, 
                    1.4, 0, 0, 0,],
    "joint_names" : ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',
                     'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0',
                     'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
                     'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0']
    }

barrett_cfg={
    "name": "barrett",
    "urdf_path" : 'hands/barret_hand/barret_hand_collisions_primitified.urdf',
    "src_urdf_path" : 'hands/barret_hand/barret_hand/barret_hand.urdf',
    "contact_points_path" : 'hands/barret_hand/contact_points.json',
    # "default_pos" : [
    #         3.1416, 2.4435, 0.8378, 3.1416, 2.4435, 0.8378, 2.4435, 0.8378 
    #     ],
    "close_pos" : [-1, -1, 2.4, 2.4, 2.4, 0.83, 0.83, 0.83 ],
    "default_pos" : [
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 
        ],
    "joint_names" : [
            'bh282_j00', 'bh282_j01', 'bh282_j02', 'bh282_j10', 'bh282_j11', 'bh282_j12', "bh282_j21", "bh282_j22", 
        ]
    }

franka_cfg={
    "urdf_path":'hands/franka_hand/franka.urdf',
    "contact_points_path":'hands/franka_hand/contact_points.json', 
    "default_pos": [0.02, 0.02],
    "joint_names" : ['left_finger', 'right_finger']
    }

shadow_cfg={
    "xml_path":'hands/shadow_hand/shadow_hand_wrist_free.xml',
    "contact_points_path":'hands/shadow_hand/contact_points.json', 
    "default_pos": [0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0],
    "joint_name" : [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
}