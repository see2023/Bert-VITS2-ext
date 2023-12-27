'''
52 index  
https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation

'''

'''
61 index send to unreal live link 
https://docs.unrealengine.com/5.3/en-US/API/Runtime/AugmentedReality/EARFaceBlendShape/
 1   EyeBlinkLeft,
 2   EyeLookDownLeft,
 3   EyeLookInLeft,
 4   EyeLookOutLeft,
 5   EyeLookUpLeft,
 6   EyeSquintLeft,
 7   EyeWideLeft,
 8   EyeBlinkRight,
 9   EyeLookDownRight,
 10   EyeLookInRight,
 11   EyeLookOutRight,
 12   EyeLookUpRight,
 13   EyeSquintRight,
 14   EyeWideRight,
 15   JawForward,
 16   JawLeft,
 17   JawRight,
 18   JawOpen,
 19   MouthClose,
 20   MouthFunnel,
 21   MouthPucker,
 22   MouthLeft,
 23   MouthRight,
 24   MouthSmileLeft,
 25   MouthSmileRight,
 26   MouthFrownLeft,
 27   MouthFrownRight,
 28   MouthDimpleLeft,
 29   MouthDimpleRight,
 30   MouthStretchLeft,
 31   MouthStretchRight,
 32   MouthRollLower,
 33   MouthRollUpper,
 34   MouthShrugLower,
 35   MouthShrugUpper,
 36   MouthPressLeft,
 37   MouthPressRight,
 38   MouthLowerDownLeft,
 39   MouthLowerDownRight,
 40   MouthUpperUpLeft,
 41   MouthUpperUpRight,
 42   BrowDownLeft,
 43   BrowDownRight,
 44   BrowInnerUp,
 45   BrowOuterUpLeft,
 46   BrowOuterUpRight,
 47   CheekPuff,
 48   CheekSquintLeft,
 49   CheekSquintRight,
 50   NoseSneerLeft,
 51   NoseSneerRight,
 52   TongueOut,
 53   HeadYaw,
 54   HeadPitch,
 55   HeadRoll,
 56   LeftEyeYaw,
 57   LeftEyePitch,
 58   LeftEyeRoll,
 59   RightEyeYaw,
 60   RightEyePitch,
 61   RightEyeRoll,

'''

import numpy as np

ARKIT_COUNT = 61
VALID_ARKIT_COUNT = 52
# 合法值在 0-1 之间的下标(序号-1), 名字中没有left right up down的表情
g_positive_index = [14, 17, 18, 19, 20, 46, 51]
g_max_value_groups = [
    [15, 16],
    [21, 22],
    [23, 24],
    [25, 26],
    [27, 28],
    [29, 30],
    [35, 36],
    [37, 38],
    [39, 40],
    [41, 42],
    [44, 45],
    [47, 48],
    [49, 50],
]


def map_arkit_values(bs_weight_arkit, mirror=True):
    # input: n * 116 float array
    weights = np.zeros((bs_weight_arkit.shape[0], ARKIT_COUNT))
    for r in range(bs_weight_arkit.shape[0]):
        for i in range(VALID_ARKIT_COUNT):
            weights[r, i] = bs_weight_arkit[r, i]

            # 从 g_max_value_groups 找出每组中最大的值，然后覆盖其他较小的值
            if mirror:
                for g in g_max_value_groups:
                    v = 0
                    for j in g:
                        if bs_weight_arkit[r, j] > v:
                            v = bs_weight_arkit[r, j]
                    for j in g:
                        weights[r, j] = v

            # weights[r, i] = weights[r, i] * 2

            jaw_scale = 1
            mouth_scale = 1
            head_scale = 0
            if i == 0:  # eyeBlinkLeft
                weights[r, i] = weights[r, i] * 1
            elif i == 17:  # jawOpen
                weights[r, i] = weights[r, i] * jaw_scale
            elif i == 18: # mouthClose
                weights[r, i] = weights[r, i] * mouth_scale
            elif i == 19: # mouthFunnel
                weights[r, i] = weights[r, i] * mouth_scale
            elif i == 20: # mouthPucker
                weights[r, i] = weights[r, i] * mouth_scale
            elif i == 21: # mouthLeft
                weights[r, i] = weights[r, i] * mouth_scale
            elif i == 22: # mouthRight
                weights[r, i] = weights[r, i] * mouth_scale
            elif i == 52: # headYaw
                weights[r, i] = weights[r, i] * head_scale
            elif i == 53: # headPitch
                weights[r, i] = weights[r, i] * head_scale
            elif i == 54: # headRoll
                weights[r, i] = weights[r, i] * head_scale

            if weights[r, i] > 1:
                weights[r, i] = 1
            if weights[r, i] < -1:
                weights[r, i] = -1
            if i in g_positive_index and weights[r, i] < 0:
                weights[r, i] = 0
        
        # weights[r] = np.convolve(weights[r], np.ones(5)/5, mode='same')


        # jawOpen * 1.5, make <=1
        # weights[r, 17] = min(weights[r, 17] * 5, 1.0)
    return weights
