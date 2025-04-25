import numpy as np
import cv2
import os
import sys
from scipy.spatial.transform import Rotation as R
import pandas as pd

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
        _, _, T_flange = robot.forward_kinematics(joint_angles)
        T_flange_list.append(T_flange)  # 保存到列表

    return T_flange_list

def tool_pos_to_transform_matrix(tool_pos_list):
    """
    将 tool_pos_list 中的位姿数据转换为变换矩阵(xyz 内旋顺序)
    
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

    
    return Tool_transform_matrix_list

def calibrate_AX_equals_YB(A_list, B_list, tsai_weight=0.5, park_weight=0.5):
    """
    使用 OpenCV 的 calibrateRobotWorldHandEye 函数求解 AX = YB 标定问题.
    同时使用 TSAI 和 PARK 两种方法，并将结果按权重融合。

    该函数用于确定两个未知变换：
    1. 工具坐标系相对于机器人法兰坐标系的变换 (X: Flange -> Tool) 
    2. 机器人基座坐标系相对于激光跟踪仪坐标系的变换 (Y: Base -> Laser)
    3. 激光跟踪仪坐标系相对于机器人基座坐标系的变换 (Y_inv: Laser -> Base)

    方程为: A * X = Y * B

    其中:
        A: Base -> Flange (机器人法兰相对于基座的位姿，来自机器人正解)
        B: Laser -> Tool (工具末端在激光跟踪仪坐标系下的位姿，来自外部测量)
        X: Base -> Laser (待求解的基座到激光跟踪仪的变换)
        Y: Flange -> Tool (待求解的工具变换)
        Y_inv: Laser -> Base (Y的逆变换，激光跟踪仪到基座的变换)

    参数:
        A_list: 包含多个 A 变换矩阵 (4x4 numpy array) 的列表 (Base -> Flange)
        B_list: 包含多个 B 变换矩阵 (4x4 numpy array) 的列表 (Laser -> Tool)
        tsai_weight: TSAI 方法的权重，范围 [0,1]
        park_weight: PARK 方法的权重，范围 [0,1]

    返回:
        X: 求解得到的 Flange -> Tool 变换矩阵 (4x4 numpy array)
        Y: 求解得到的 Base -> Laser 变换矩阵 (4x4 numpy array)
        Y_inv: 求解得到的 Laser -> Base 变换矩阵，即Y的逆矩阵 (4x4 numpy array)
    """
    if len(A_list) != len(B_list) or len(A_list) < 3:
        raise ValueError("输入列表长度必须相同且至少为 3 (建议更多组非共面/共线的运动)")

    # # 验证权重参数
    # if tsai_weight < 0 or tsai_weight > 1 or park_weight < 0 or park_weight > 1:
    #     raise ValueError("权重参数必须在 [0,1] 范围内")
    
    # # 确保权重和为1
    # weight_sum = tsai_weight + park_weight
    # if abs(weight_sum - 1.0) > 1e-6:  # 允许一点浮点误差
    #     tsai_weight = tsai_weight / weight_sum
    #     park_weight = park_weight / weight_sum
    #     print(f"警告: 权重和不为1，已自动归一化为 TSAI: {tsai_weight:.4f}, PARK: {park_weight:.4f}")

    # 分解 A (Base -> Flange)
    R_base2gripper = [A[:3, :3] for A in A_list]
    t_base2gripper = [A[:3, 3].reshape(3, 1) for A in A_list]

    # 分解 B (Laser -> Tool)
    R_world2tool = [B[:3, :3] for B in B_list] # 使用 B 列表
    t_world2tool = [B[:3, 3].reshape(3, 1) for B in B_list] # 使用 B 列表

    # 使用 TSAI 方法
    R_base2world_tsai, t_base2world_tsai, R_gripper2tool_tsai, t_gripper2tool_tsai = cv2.calibrateRobotWorldHandEye(
        R_world2cam=R_world2tool,   
        t_world2cam=t_world2tool,   
        R_base2gripper=R_base2gripper, 
        t_base2gripper=t_base2gripper,  
        method=cv2.CALIB_HAND_EYE_TSAI 
    )

    # 使用 PARK 方法
    R_base2world_park, t_base2world_park, R_gripper2tool_park, t_gripper2tool_park = cv2.calibrateRobotWorldHandEye(
        R_world2cam=R_world2tool,   
        t_world2cam=t_world2tool,   
        R_base2gripper=R_base2gripper, 
        t_base2gripper=t_base2gripper,  
        method=cv2.CALIB_HAND_EYE_PARK 
    )

    # 将旋转矩阵转换为四元数进行加权融合
    rot_base2world_tsai = R.from_matrix(R_base2world_tsai)
    rot_base2world_park = R.from_matrix(R_base2world_park)
    rot_gripper2tool_tsai = R.from_matrix(R_gripper2tool_tsai)
    rot_gripper2tool_park = R.from_matrix(R_gripper2tool_park)

    quat_base2world_tsai = rot_base2world_tsai.as_quat()
    quat_base2world_park = rot_base2world_park.as_quat()
    quat_gripper2tool_tsai = rot_gripper2tool_tsai.as_quat()
    quat_gripper2tool_park = rot_gripper2tool_park.as_quat()

    # 确保四元数方向一致性（避免插值时绕远路）
    if np.dot(quat_base2world_tsai, quat_base2world_park) < 0:
        quat_base2world_park = -quat_base2world_park
    if np.dot(quat_gripper2tool_tsai, quat_gripper2tool_park) < 0:
        quat_gripper2tool_park = -quat_gripper2tool_park

    # 加权融合四元数
    quat_base2world = slerp_quaternion(quat_base2world_tsai, quat_base2world_park, park_weight)
    quat_gripper2tool = slerp_quaternion(quat_gripper2tool_tsai, quat_gripper2tool_park, park_weight)

    # 加权融合平移向量
    t_base2world = tsai_weight * t_base2world_tsai + park_weight * t_base2world_park
    t_gripper2tool = tsai_weight * t_gripper2tool_tsai + park_weight * t_gripper2tool_park

    # 将四元数转回旋转矩阵
    R_base2world = R.from_quat(quat_base2world).as_matrix()
    R_gripper2tool = R.from_quat(quat_gripper2tool).as_matrix()

    # 组合 X (Flange -> Tool)
    X = np.eye(4)
    X[:3, :3] = R_gripper2tool
    X[:3, 3] = t_gripper2tool.flatten()

    # 组合 Y (Base -> Laser/World)
    Y = np.eye(4)
    Y[:3, :3] = R_base2world
    Y[:3, 3] = t_base2world.flatten()
    
    # 计算 Y_inv (Laser -> Base)，即 Y 的逆矩阵
    Y_inv = np.linalg.inv(Y)

    # 创建 results 目录（如果不存在）
    os.makedirs('results', exist_ok=True)
    
    # 准备写入文件的内容
    with open('results/calibration.txt', 'w', encoding='utf-8') as f:
        f.write(f"--- AX=YB 标定结果 (TSAI 权重={tsai_weight:.4f}, PARK 权重={park_weight:.4f}) ---\n\n")
        
        # 打印单独使用 TSAI 的结果
        X_tsai = np.eye(4)
        X_tsai[:3, :3] = R_gripper2tool_tsai
        X_tsai[:3, 3] = t_gripper2tool_tsai.flatten()
        
        Y_tsai = np.eye(4)
        Y_tsai[:3, :3] = R_base2world_tsai
        Y_tsai[:3, 3] = t_base2world_tsai.flatten()
        
        Y_inv_tsai = np.linalg.inv(Y_tsai)
        
        f.write("TSAI 方法结果:\n")
        write_transform_info(f, X_tsai, "X (Base -> Laser)")
        write_transform_info(f, Y_tsai, "Y (Flange -> Tool)")
        write_transform_info(f, Y_inv_tsai, "Y_inv (Tool -> Flange)")
        f.write("\n")
        
        # 打印单独使用 PARK 的结果
        X_park = np.eye(4)
        X_park[:3, :3] = R_gripper2tool_park
        X_park[:3, 3] = t_gripper2tool_park.flatten()
        
        Y_park = np.eye(4)
        Y_park[:3, :3] = R_base2world_park
        Y_park[:3, 3] = t_base2world_park.flatten()
        
        Y_inv_park = np.linalg.inv(Y_park)
        
        f.write("PARK 方法结果:\n")
        write_transform_info(f, X_park, "X (Base -> Laser)")
        write_transform_info(f, Y_park, "Y (Flange -> Tool)")
        write_transform_info(f, Y_inv_park, "Y_inv (Tool -> Flange)")
        f.write("\n")
        
        # 打印融合后的结果
        f.write("融合后的最终结果:\n")
        write_transform_info(f, X, "X (Base -> Laser)")
        write_transform_info(f, Y, "Y (Flange -> Tool)")
        write_transform_info(f, Y_inv, "Y_inv (Tool -> Flange)")

    # 在控制台输出融合结果
    print(f"\n--- AX=YB 标定结果 (TSAI 权重={tsai_weight:.4f}, PARK 权重={park_weight:.4f}) ---")
    X_pos = X[:3, 3]
    X_rot = R.from_matrix(X[:3, :3])
    X_quat = X_rot.as_quat()
    print("X (Base -> Laser):")
    print("  矩阵:")
    print(X)
    print(f"  平移 (x, y, z): {X_pos[0]:.6f}, {X_pos[1]:.6f}, {X_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {X_quat[0]:.6f}, {X_quat[1]:.6f}, {X_quat[2]:.6f}, {X_quat[3]:.6f}")
    
    Y_pos = Y[:3, 3]
    Y_rot = R.from_matrix(Y[:3, :3])
    Y_quat = Y_rot.as_quat()
    print("\nY (Flange -> Tool):")
    print("  矩阵:")
    print(Y)
    print(f"  平移 (x, y, z): {Y_pos[0]:.6f}, {Y_pos[1]:.6f}, {Y_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {Y_quat[0]:.6f}, {Y_quat[1]:.6f}, {Y_quat[2]:.6f}, {Y_quat[3]:.6f}")
    
    Y_inv_pos = Y_inv[:3, 3]
    Y_inv_rot = R.from_matrix(Y_inv[:3, :3])
    Y_inv_quat = Y_inv_rot.as_quat()
    print("\nY_inv (Tool -> Flange):")
    print("  矩阵:")
    print(Y_inv)
    print(f"  平移 (x, y, z): {Y_inv_pos[0]:.6f}, {Y_inv_pos[1]:.6f}, {Y_inv_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {Y_inv_quat[0]:.6f}, {Y_inv_quat[1]:.6f}, {Y_inv_quat[2]:.6f}, {Y_inv_quat[3]:.6f}")

    return X, Y, Y_inv

def write_transform_info(f, transform, name):
    """辅助函数：将变换矩阵信息写入文件"""
    pos = transform[:3, 3]
    rot = R.from_matrix(transform[:3, :3])
    quat = rot.as_quat()
    
    f.write(f"{name}:\n")
    f.write("  矩阵:\n")
    f.write(str(transform) + "\n")
    f.write(f"  平移 (x, y, z): {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}\n")
    f.write(f"  旋转 (四元数 x, y, z, w): {quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}\n\n")

def slerp_quaternion(q1, q2, t):
    """
    球面线性插值两个四元数
    
    参数:
        q1: 第一个四元数 [x,y,z,w]
        q2: 第二个四元数 [x,y,z,w]
        t: 插值参数 [0,1]，t=0 返回 q1，t=1 返回 q2
        
    返回:
        插值后的四元数
    """
    # 确保输入为单位四元数
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算四元数的点积
    dot = np.sum(q1 * q2)
    
    # 如果点积为负，取其中一个四元数的负数以确保最短路径插值
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # 如果四元数几乎相同，线性插值即可
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
        
    # 计算夹角
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    # 执行 SLERP
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2

# *测试
if __name__ == "__main__":
    # 读取关节角度数据
    df_joint_angles = pd.read_csv('data/joint_angle.csv', header=None, skiprows=1)
    joint_angles_list = df_joint_angles.iloc[:, :6].values.tolist()

    # 读取工具位姿数据
    df_tool_pos = pd.read_csv('data/tool_pos_laser.csv', header=None, skiprows=1)
    tool_pos_list = df_tool_pos.iloc[:, :6].values.tolist()

    # 调用函数计算并保存 T_flange
    T_flange_list = calculate_T_flange(joint_angles_list)

    # 转换为变换矩阵
    Tool_transform_matrix_list = tool_pos_to_transform_matrix(tool_pos_list)

    # --------------------------------------------------
    #    调用 AX=YB 标定函数
    # --------------------------------------------------
    X_flange2tool, Y_base2Laser, Y_inv_laser2base = calibrate_AX_equals_YB(
        T_flange_list, 
        Tool_transform_matrix_list,
        tsai_weight=-0.19626782121420666,
        park_weight=1.1991339962029677

    )
