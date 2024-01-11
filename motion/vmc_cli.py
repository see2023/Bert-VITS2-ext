#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Basic VMC protocol example."""
import sys
from typing import Any
import numpy as np
from numpy import cross, dot
from numpy.linalg import norm
import time as systime
from math import (radians, sin)
# OSC
from vmcp.osc import OSC
from vmcp.osc.typing import Message
from vmcp.osc.backend.osc4py3 import as_eventloop as backend
# VMC protocol layer
from vmcp.typing import (
    CoordinateVector,
    Quaternion,
    Bone,
    ModelState,
    Timestamp
)
from vmcp.protocol import (
    root_transform,
    bone_transform,
    state,
    time
)

'''
# bones:
# from https://github.com/OpenMotionLab/MotionGPT
humanml3d_joints = [
    "root",
    "RH",
    "LH",
    "BP",
    "RK",
    "LK",
    "BT",
    "RMrot",
    "LMrot",
    "BLN",
    "RF",
    "LF",
    "BMN",
    "RSI",
    "LSI",
    "BUN",
    "RS",
    "LS",
    "RE",
    "LE",
    "RW",
    "LW",
]

smplnh_joints = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
    '''

g_smpl_bones  = (
    Bone.HIPS,
    Bone.LEFT_UPPER_LEG,
    Bone.RIGHT_UPPER_LEG,
    Bone.SPINE,
    Bone.LEFT_LOWER_LEG,
    Bone.RIGHT_LOWER_LEG,
    Bone.CHEST,
    Bone.LEFT_FOOT,
    Bone.RIGHT_FOOT,
    Bone.UPPER_CHEST,
    Bone.LEFT_TOES,
    Bone.RIGHT_TOES,
    Bone.NECK,
    Bone.LEFT_SHOULDER,
    Bone.RIGHT_SHOULDER,
    Bone.HEAD,
    Bone.LEFT_UPPER_ARM,
    Bone.RIGHT_UPPER_ARM,
    Bone.LEFT_LOWER_ARM,
    Bone.RIGHT_LOWER_ARM,
    Bone.LEFT_HAND,
    Bone.RIGHT_HAND,
)

g_humanml3d_bones = (
    Bone.HIPS,
    Bone.RIGHT_UPPER_LEG,
    Bone.LEFT_UPPER_LEG,
    Bone.SPINE,
    Bone.RIGHT_LOWER_LEG,
    Bone.LEFT_LOWER_LEG,
    Bone.CHEST,
    Bone.RIGHT_FOOT,
    Bone.LEFT_FOOT,
    Bone.UPPER_CHEST,
    Bone.RIGHT_TOES,
    Bone.LEFT_TOES,
    Bone.NECK,
    Bone.RIGHT_SHOULDER,
    Bone.LEFT_SHOULDER,
    Bone.HEAD,
    Bone.RIGHT_UPPER_ARM,
    Bone.LEFT_UPPER_ARM,
    Bone.RIGHT_LOWER_ARM,
    Bone.LEFT_LOWER_ARM,
    Bone.RIGHT_HAND,
    Bone.LEFT_HAND,
)

humanml3d_kinematic_tree = [
    # [0, 3, 6, 9, 12, 15],  # body
    [9, 14, 17, 19, 21],  # right arm
    [9, 13, 16, 18, 20],  # left arm
    [0, 2, 5, 8, 11],  # right leg
    [0, 1, 4, 7, 10],  # left leg
]  


SMPL_PARRENTS = [
    -1,  # 0
    0,  # 1
    0,  # 2
    0,  # 3
    1,  # 4
    2,  # 5
    3,  # 6
    4,  # 7
    5,  # 8
    6,  # 9
    7,  # 10
    8,  # 11
    9,  # 12
    9,  # 13
    9,  # 14
    12,  # 15
    13,  # 16
    14,  # 17
    16,  # 18
    17,  # 19
    18,  # 20
    19,  # 21
    20,  # 22
    21,  # 23
]

# relative position
# from https://github.com/HAL9HARUKU
t_pose_bone_position_map = {
    Bone.HIPS: (0.0003016567789018154, 0.8938107490539551, 0.006238983478397131),
    Bone.LEFT_UPPER_LEG: (-0.07712238281965256, -0.039701998233795166, -0.004908805713057518),
    Bone.RIGHT_UPPER_LEG: (0.07712236791849136, -0.03970217704772949, -0.004908811300992966),
    Bone.SPINE: (-6.781192496418953e-09, 0.052127957344055176, 0.009726937860250473),
    Bone.LEFT_LOWER_LEG: (0.019634976983070374, -0.342479407787323, -0.006634509190917015),
    Bone.RIGHT_LOWER_LEG: (-0.019635319709777832, -0.34247899055480957, -0.0066344477236270905),
    Bone.CHEST: (-1.4493707567453384e-08, 0.1108858585357666, 0.014510296285152435),
    Bone.LEFT_FOOT: (0.010010819882154465, -0.40157943964004517, -0.020943094044923782),
    Bone.RIGHT_FOOT: (-0.010011687874794006, -0.40157967805862427, -0.02094341814517975),
    Bone.UPPER_CHEST: (1.2514647096395493e-08, 0.12428486347198486, -0.013312757015228271),
    Bone.LEFT_TOES: (-0.0014426521956920624, -0.06444710493087769, 0.1105375587940216),
    Bone.RIGHT_TOES: (0.001442044973373413, -0.06444692611694336, 0.11053761839866638),
    Bone.NECK: (1.0564690455794334e-08, 0.11461520195007324, -0.0333956740796566),
    Bone.LEFT_SHOULDER: (-0.02238563261926174, 0.08726394176483154, -0.02744283899664879),
    Bone.RIGHT_SHOULDER: (0.022385617718100548, 0.08726680278778076, -0.027442801743745804),
    Bone.HEAD: (-2.2497260943055153e-08, 0.07376956939697266, 0.009335324168205261),
    Bone.LEFT_UPPER_ARM: (-0.08629470318555832, -0.014869093894958496, 0.005363747477531433),
    Bone.RIGHT_UPPER_ARM: (0.08629446476697922, -0.0148698091506958, 0.005363717675209045),
    Bone.LEFT_LOWER_ARM: (-0.21016523241996765, -0.009785890579223633, 0.001763179898262024),
    Bone.RIGHT_LOWER_ARM: (0.21016687154769897, -0.0097883939743042, 0.0017633810639381409),
    Bone.LEFT_HAND: (-0.20475271344184875, -0.0004464387893676758, 0.016901567578315735),
    Bone.RIGHT_HAND: (0.20475167036056519, -0.00044548511505126953, 0.016901902854442596),
}

g_bone_factor_map = {
    Bone.HIPS: 1,
    Bone.LEFT_UPPER_LEG: 0.1,
    Bone.RIGHT_UPPER_LEG: 0.1,
    Bone.SPINE: 0.2,
    Bone.LEFT_LOWER_LEG: 0.5,
    Bone.RIGHT_LOWER_LEG: 0.5,
    Bone.CHEST: 0.8,
    Bone.LEFT_FOOT: 0.8,
    Bone.RIGHT_FOOT: 0.8,
    Bone.UPPER_CHEST: 0.8,
    Bone.LEFT_TOES: 1,
    Bone.RIGHT_TOES: 1,
    Bone.NECK: 1,
    Bone.LEFT_SHOULDER: 1,
    Bone.RIGHT_SHOULDER: 1,
    Bone.HEAD: 1,
    Bone.LEFT_UPPER_ARM: 0.9,
    Bone.RIGHT_UPPER_ARM: 0.9,
    Bone.LEFT_LOWER_ARM: 0.9,
    Bone.RIGHT_LOWER_ARM: 0.9,
    Bone.LEFT_HAND: 0.9,
    Bone.RIGHT_HAND: 0.9,
}

# 当SMPL_PARRENTS[i] 1= -1时，计算相对位置
def to_relative_position(positions):
    # new array
    relative_positions = []
    relative_positions.append(positions[0])
    for i in range(1, len(positions)):
        parrent_index = SMPL_PARRENTS[i]
        relative_positions.append(positions[i] - positions[parrent_index])
    return np.array(relative_positions).copy()

# relative position
g_humanml3d_t_pose_positions = np.array([
    t_pose_bone_position_map[bone] for bone in g_humanml3d_bones
])


def calculate_rotation_quaternion(line1, line2, bone=None, factor=1):

    # 保证line1和line2是单位向量
    line1 = line1 / norm(line1)
    line2 = line2 / norm(line2)

    # 计算点积
    cos_theta = dot(line1, line2)
    # 避免数值误差导致的问题
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = np.arccos(cos_theta) # 转动角度
    theta *= factor
    if bone in g_bone_factor_map:
        theta *= g_bone_factor_map[bone]

    # 计算叉积
    rotation_axis = cross(line1, line2)
    rotation_axis_length = norm(rotation_axis)

    if rotation_axis_length > 0:
        # 单位化旋转轴
        rotation_axis = rotation_axis / rotation_axis_length
    else:
        # 如果line1和line2是平行的，需要找到一个垂直于他们的任意向量作为旋转轴
        # 如果line1不是(1,0,0), 我们可以用(1,0,0)叉乘line1得到旋转轴；
        # 否则我们可以用(0,1,0)叉乘line1.
        if np.allclose(line1, [1.0, 0.0, 0.0]):
            rotation_axis = np.array([0.0, 1.0, 0.0])
        else:
            rotation_axis = np.cross([1.0, 0.0, 0.0], line1)

    q = np.empty(4)
    q[3] = np.cos(theta / 2.0) # 四元数实部
    q[0:3] = rotation_axis * np.sin(theta / 2.0) # 四元数虚部

    return q

# 扩展 Quaternion 类，增加共轭、norm、逆，旋转的计算
class MyQuaternion(Quaternion):
    def __init__(self, x, y, z, w):
        super().__init__(x, y, z, w)
    
    # 乘常数
    def multiply(self, v):
        return MyQuaternion(self.x * v, self.y * v, self.z * v, self.w * v)

    # 四元数的共轭（代表逆旋转）
    def quaternion_conjugate(self):
        return MyQuaternion(-self.x, -self.y, -self.z, self.w)

    # 模
    def norm(self):
        return (self.w**2 + self.x**2 + self.y**2 + self.z**2)**0.5

    # 四元数的逆
    def quaternion_inverse(self):
        return self.quaternion_conjugate().multiply( 1 / self.norm()**2 )

    # 计算line2到line3的四元数
    # self: line1 --> line2
    # q13: line1 --> line3
    def rotation_to(self, q13):
        q12_inv = self.quaternion_inverse()
        return q12_inv.multiply_by(q13)


# 计算每个位置的连线到下一段联系的转动角度的四元数
def postions_to_quaternions(positions, last_positions, bones, control_by_parrent = True):
    quaternions = []
    quaternions.append(MyQuaternion.identity())
    for i in range(1, len(positions)):
        # parrent_index = SMPL_PARRENTS[i]
        # # 上一时刻的连线
        # last_line = last_positions[i] - last_positions[parrent_index]
        # # 当前时刻的连线
        # line = positions[i] - positions[parrent_index]

        # 计算连线的转动角度的四元数
        quat = calculate_rotation_quaternion(last_positions[i], positions[i], bones[i])
        quaternions.append(MyQuaternion(*quat))

    # 从叶子节点开始，如果存在父节点，减去父节点的旋转 calculate_relative_rotation
    for i in range(len(quaternions)-1, 0, -1):
        parrent_index = SMPL_PARRENTS[i]
        if parrent_index == -1:
            continue
        quaternions[i] = quaternions[parrent_index].rotation_to(quaternions[i])
    
    if control_by_parrent:
        # 按数组 humanml3d_kinematic_tree 列出的父子关系，将子节点根据位置计算出的旋转赋值给父节点(从每个数组的第三个开始，最后一个赋值为0)
        for bone_list in humanml3d_kinematic_tree:
            for i in range(2, len(bone_list)):
                parrent_index = bone_list[i-1]
                child_index = bone_list[i]
                quaternions[parrent_index] = quaternions[child_index]
            # 将每组末尾节点的旋转赋值为0
            quaternions[bone_list[-1]] = Quaternion.identity()
        # reset head rotation
        quaternions[15] = Quaternion.identity()

    return quaternions



def gen_bone_message_tuple(bone_tuple, positions, last_positions, init_positions, is_first_frame):
    messages = ()
    quaternions = ()
    if is_first_frame:
        for (bone, position) in zip(bone_tuple, positions):
            if bone is None:
                continue
            messages += (
                Message( *bone_transform(
                    bone, 
                    # CoordinateVector.identity(), 
                    # CoordinateVector(position[0], position[1], position[2]),
                    CoordinateVector(t_pose_bone_position_map[bone][0], t_pose_bone_position_map[bone][1], t_pose_bone_position_map[bone][2]),
                    Quaternion.identity(),
                )),
            )
    else:
        # quaternions = postions_to_quaternions(positions, last_positions)
        # quaternions = postions_to_quaternions(positions, init_positions)
        quaternions = postions_to_quaternions(positions, g_humanml3d_t_pose_positions, bone_tuple)

        for (bone, position, init_position, quaternion, t_pose_position) in zip(bone_tuple, positions, init_positions, quaternions, g_humanml3d_t_pose_positions):
            if bone is None:
                continue
            # position *= 0.09
            messages += (
                Message( *bone_transform(
                    bone, 
                    # CoordinateVector.identity(), 
                    # CoordinateVector(position[0], position[1], position[2]),
                    # CoordinateVector(init_position[0], init_position[1], init_position[2]),
                    CoordinateVector(t_pose_position[0], t_pose_position[1], t_pose_position[2]),


                    # Quaternion.identity(),
                    quaternion,
                )),
            )
    return messages



global_counter = 0
global_init_postion = None
def send_osc_data(sender, cur_positions, last_positions, is_first_frame = False):
    cur_positions = to_relative_position(cur_positions)
    if not is_first_frame:
        last_positions = to_relative_position(last_positions)
    
    global global_counter
    global global_init_postion

    if is_first_frame:
        global_init_postion = cur_positions.copy()
    # data_list (22, 3)
    # print('data_list', data_list)

    y_off_set = -0.17

    root_pos = CoordinateVector.identity()
    if not is_first_frame:
        scale = 0.4
        root_pos = CoordinateVector(
            scale * (cur_positions[0][0]-global_init_postion[0][0]), 
            scale*(cur_positions[0][1]-global_init_postion[0][1]) + y_off_set,
            scale*( cur_positions[0][2]-global_init_postion[0][2])
        )

    messages = (
        Message(*root_transform(
            # CoordinateVector.identity(), 
            root_pos,
            # CoordinateVector(cur_positions[0][0], cur_positions[0][1], cur_positions[0][2]),
            Quaternion.identity()
        )),
    )
    global_counter += 1

    bones  = g_humanml3d_bones

    messages += gen_bone_message_tuple(
        bones,
        cur_positions,
        last_positions,
        global_init_postion,
        is_first_frame,
    )
    messages += (
                Message(*state(ModelState.LOADED)),
                Message(*time(Timestamp())),
    )
    sender.send(messages)


def send_montions(file_paths):
    #np读取文件列表，在第二维度上合并
    print('file count:', len(file_paths))
    bs = np.concatenate([np.load(file_path, allow_pickle=True) for file_path in file_paths], axis=1)
    # 最后一维的第一个元素取反，坐标系单轴镜像
    # bs[:, :, :, 0] *= -1
    print(bs.shape)
    # (1, 192, 22, 3)
    # 在 axis=1上线性插值成两倍数据 (1, 384, 22, 3)
    new_bs = np.zeros((bs.shape[0], bs.shape[1]*2, bs.shape[2], bs.shape[3]))
    for i in range(0, bs.shape[0]):
        for j in range(0, bs.shape[2]):
            for k in range(0, bs.shape[3]):
                line_data = bs[i, :, j, k]
                line_data = np.interp(np.arange(0, len(line_data), 0.5), np.arange(0, len(line_data)), line_data)
                new_bs[i, :, j, k] = line_data
    bs = new_bs
    print(bs.shape)
    fps = 30

    # 将上述时序数据平滑 0-t(192)
    if True:
        # 将第二维放到最末尾: 0, 1, 2, 3 --> 0, 2, 3, 1
        bs = np.swapaxes(bs, 1, 2)
        bs = np.swapaxes(bs, 2, 3)
        for i in range(0, bs.shape[1]):
            for j in range(0, bs.shape[2]):
                line_data = bs[0][i][j] #.copy()
                line_data_padded = np.pad(line_data, 4, mode='edge')
                line_data_smoothed = np.convolve(line_data_padded, np.ones(9)/9, mode='valid')
                bs[0][i][j] = line_data_smoothed
        bs = np.swapaxes(bs, 3, 2)
        bs = np.swapaxes(bs, 2, 1)

    try:
        osc = OSC(backend)
        with osc.open():
            # Sender
            print("Sending...")
            sender = osc.create_sender("192.168.17.141", 39539, "sender1").open()
            # send data from bs every 1/fps seconds
            total_frames = bs.shape[1]
            start_time = systime.time()
            for i in range(0, bs.shape[1]):
            # for i in range(0, 1):
                now = systime.time()
                next_send_time = start_time + (i+1) / fps
                sleep_time = next_send_time - now
                if i % 30 == 0:
                    print('now', now, 'next_send_time', next_send_time, 'sleep_time', sleep_time)
                if sleep_time > 0:
                    systime.sleep(sleep_time)
                send_osc_data(sender, bs[0][i], bs[0][i-1],  i == 0)
                osc.run()

    except KeyboardInterrupt:
        print("Canceled.")
    finally:
        osc.close()

# get input file path
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('python vmc_cli.py np_file_path...')
        exit(1)

    np_file_paths = sys.argv[1:] 
    send_montions(np_file_paths)
