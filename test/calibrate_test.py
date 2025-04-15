import numpy as np
import cv2
import os
import sys
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from forward_kinematics import RobotKinematics

# 禁用科学计数法，使输出更易读
np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

def calculate_T_flange(joint_angles_list):
    """
    计算并保存多组关节角度的正运动学变换矩阵 T_flange
    
    参数:
        joint_angles_list: 多组关节角度的二维列表
    
    返回:
        T_flange_list: 包含所有 T_flange 的列表
    """
    # 创建机器人运动学实例
    robot = RobotKinematics()
    
    # 初始化空列表
    T_flange_list = []
    
    # 遍历计算每组关节角度的正运动学
    for i, joint_angles in enumerate(joint_angles_list):
        position, quaternion, T_flange = robot.forward_kinematics(joint_angles)
        T_flange_list.append(T_flange)  # 保存到列表

    print("\n所有变换矩阵 T_flange:")
    for idx, T in enumerate(T_flange_list):
        print(f"第 {idx+1} 组 T_flange:")
        print(T)
    
    return T_flange_list

def tool_pos_to_transform_matrix(tool_pos_list):
    """
    将 tool_pos_list 中的位姿数据转换为变换矩阵(XYZ 外旋顺序)
    
    参数:
        tool_pos_list: 包含多组位姿的二维列表，每组为 [x, y, z, rx, ry, rz]
    
    返回:
        Tool_transform_matrix_list: 包含所有变换矩阵的列表
    """
    Tool_transform_matrix_list = []
    
    for pos in tool_pos_list:
        x, y, z, rx, ry, rz = pos
        
        # 1. 处理旋转部分（xyz 内旋）
        rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        rotation_matrix = rotation.as_matrix()
        
        # 2. 处理平移部分
        translation = np.array([x, y, z])
        
        # 3. 组合为 4x4 变换矩阵
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        
        Tool_transform_matrix_list.append(T)
        # 打印结果
        print("\n所有工具位姿的变换矩阵:")
        for idx, T in enumerate(Tool_transform_matrix_list):
            print(f"第 {idx+1} 组 Tool_transform_matrix:")
            print(T)

    
    return Tool_transform_matrix_list

def calibrate_AX_equals_YB(A_list, B_list):
    """
    使用 OpenCV 的 calibrateRobotWorldHandEye 函数求解 AX = YB 标定问题.

    该函数用于确定两个未知变换：
    1. 工具坐标系相对于机器人法兰坐标系的变换 (Y: Flange -> Tool) 
    2. 机器人基座坐标系相对于世界坐标系/激光跟踪仪坐标系的变换 (X: Base -> Laser)

    方程为: A * X = Y * B

    其中:
        A: Base -> Flange (机器人法兰相对于基座的位姿，来自机器人正解)
        B: Laser -> Tool (工具末端在激光跟踪仪坐标系下的位姿，来自外部测量)
        X: Flange -> Tool (待求解的工具变换)
        Y: Base -> Laser (待求解的基座到激光跟踪仪的变换)

    参数:
        A_list: 包含多个 A 变换矩阵 (4x4 numpy array) 的列表 (Base -> Flange)
        B_list: 包含多个 B 变换矩阵 (4x4 numpy array) 的列表 (Laser -> Tool)

    返回:
        X: 求解得到的 Flange -> Tool 变换矩阵 (4x4 numpy array)
        Y: 求解得到的 Base -> Laser 变换矩阵 (4x4 numpy array)
    """
    if len(A_list) != len(B_list) or len(A_list) < 3:
        raise ValueError("输入列表长度必须相同且至少为 3 (建议更多组非共面/共线的运动)")

    # 分解 A (Base -> Flange)
    R_base2gripper = [A[:3, :3] for A in A_list]
    t_base2gripper = [A[:3, 3].reshape(3, 1) for A in A_list]

    # 分解 B (Laser -> Tool)
    R_world2tool = [B[:3, :3] for B in B_list] # 使用 B 列表
    t_world2tool = [B[:3, 3].reshape(3, 1) for B in B_list] # 使用 B 列表

    # 调用 OpenCV 的标准 AX=YB 求解器: cv2.calibrateRobotWorldHandEye
    # R_world2cam, t_world2cam <- 来自 B_list (Laser -> Tool)
    # R_base2gripper, t_base2gripper <- 来自 A_list (Base -> Flange)
    R_base2world, t_base2world, R_gripper2tool, t_gripper2tool = cv2.calibrateRobotWorldHandEye(
        R_world2cam=R_world2tool,   # 使用从 B 分解出的旋转
        t_world2cam=t_world2tool,   # 使用从 B 分解出的平移
        R_base2gripper=R_base2gripper, # 使用从 A 分解出的旋转
        t_base2gripper=t_base2gripper  # 使用从 A 分解出的平移
    )

    # 组合 X (Flange -> Tool)
    X = np.eye(4)
    X[:3, :3] = R_gripper2tool
    X[:3, 3] = t_gripper2tool.flatten()

    # 组合 Y (Base -> Laser/World)
    # 注意：返回值是 R_base2world, t_base2world，但在我们的场景中它代表 Base -> Laser
    Y = np.eye(4)
    Y[:3, :3] = R_base2world
    Y[:3, 3] = t_base2world.flatten()

    # 在函数内部打印结果
    print("\n--- AX=YB 标定结果 (使用 cv2.calibrateRobotWorldHandEye) ---")

    # 提取并打印 X (Base -> Laser) 的 xyz 和四元数 (x, y, z, w)
    X_pos = X[:3, 3]
    X_rot = R.from_matrix(X[:3, :3])
    X_quat = X_rot.as_quat() # 获取 [x, y, z, w] 格式的四元数
    print("X (Base -> Laser):")
    print("  矩阵:")
    print(X)
    print(f"  平移 (x, y, z): {X_pos[0]:.6f}, {X_pos[1]:.6f}, {X_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {X_quat[0]:.6f}, {X_quat[1]:.6f}, {X_quat[2]:.6f}, {X_quat[3]:.6f}")

    # 提取并打印 Y (Flange -> Tool) 的 xyz 和四元数 (x, y, z, w)
    Y_pos = Y[:3, 3]
    Y_rot = R.from_matrix(Y[:3, :3])
    Y_quat = Y_rot.as_quat() # 获取 [x, y, z, w] 格式的四元数
    print("\nY (Flange -> Tool):")
    print("  矩阵:")
    print(Y)
    print(f"  平移 (x, y, z): {Y_pos[0]:.6f}, {Y_pos[1]:.6f}, {Y_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {Y_quat[0]:.6f}, {Y_quat[1]:.6f}, {Y_quat[2]:.6f}, {Y_quat[3]:.6f}")

    return X, Y

# TODO: 测试标定
# 输入多组关节角度
joint_angles_list = [
    [2.636115843805, 29.5175455057437, 6.10038158777613, 34.7483282137072, -47.3762053517954, 35.7916251381285],
    [21.9332955173743, 1.47737168374035, 16.6041763862692, 77.4494485491442, -56.6181359662757, -13.918496932557],
    [11.9416940707981, 32.0408374574178, -14.3784399916024, 58.7160061596109, -35.1629125653601, -3.249745424493],
    [25.7872733836713, 2.91726236892728, 47.4129707149368, 54.2643359054748, -70.6869875024335, 24.4322224215242],
    [22.7893069086801, 27.9177622958598, -27.5333193370876, 92.8322938195731, -50.978765223296, -30.4885413174039],
]

tool_pos_list = [
    [3217.5715,1957.9307,842.7881,166.7134,-63.6497,-24.6327],
    [3511.9021,2236.7876,1115.9663,179.8692,-55.0875,-46.6583],
    [3417.1236,1854.6945,1036.6267,176.2284,-52.0739,-28.7116],
    [3574.5159,2329.1989,749.7792,169.9358,-56.1041,-30.5712],
    [3583.595,1969.3769,1255.3038,-174.8057,-61.8494,-46.6957],
]
# 调用函数计算并保存 T_flange
T_flange_list = calculate_T_flange(joint_angles_list)

# 转换为变换矩阵
Tool_transform_matrix_list = tool_pos_to_transform_matrix(tool_pos_list)

# --------------------------------------------------
#    调用 AX=YB 标定函数
# --------------------------------------------------
try:
    X_flange2tool, Y_base2Laser = calibrate_AX_equals_YB(T_flange_list, Tool_transform_matrix_list)

except ValueError as e:
    print(f"标定错误: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")



