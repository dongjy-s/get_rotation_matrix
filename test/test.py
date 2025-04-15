# test/test.py
import numpy as np
import sys
import os

# 将父目录（项目根目录）添加到 Python 路径中
# 这样才能找到 forward_kinematics 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 禁用科学计数法，使输出更易读
np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

# 导入机器人运动学类
try:
    from forward_kinematics import RobotKinematics
except ImportError:
    print("错误：无法导入 forward_kinematics 模块。")
    print("请确保 forward_kinematics.py 文件位于项目的根目录下，并且包含 RobotKinematics 类。")
    sys.exit(1)

# 创建运动学实例
robot = RobotKinematics()

# 定义一组示例关节角度 (单位：度)
# 你可以修改这些值来测试不同的姿态
joint_angles_deg = np.array([7.06126418859886,18.8924281373764,42.2381279898771,17.4298727548666,-70.1243652930826,53.5458812983743])

print(f"输入关节角度 (度):   {joint_angles_deg}")


# 调用正运动学计算
try:
    _, _, T_flange_to_base = robot.forward_kinematics(joint_angles_deg)

    print("\n正运动学计算结果 (末端法兰相对于基座的变换矩阵 T_flange2base):")
    print(T_flange_to_base)

    # --- 计算工具相对于基座的位姿 ---
    print("\n计算工具相对于基座的位姿 (T_tool2base = T_flange2base * T_tool2flange):")

    # 1. 定义工具相对于法兰的位姿 (T_tool2flange)
    tool_pos_in_flange = np.array([0.1731, 1.1801, 238.3535]) # X, Y, Z in mm
    tool_quat_in_flange = np.array([0.4961, 0.5031, 0.505, 0.4957]) # Rx, Ry, Rz, W

    # 确保四元数归一化 (可选但推荐)
    tool_quat_in_flange /= np.linalg.norm(tool_quat_in_flange)

    print(f"  工具相对于法兰的位置 (X,Y,Z): {tool_pos_in_flange}")
    print(f"  工具相对于法兰的姿态 (四元数 xyzw): {tool_quat_in_flange}")

    # 2. 从四元数创建旋转矩阵
    try:
        from scipy.spatial.transform import Rotation as R
        tool_rot_in_flange = R.from_quat(tool_quat_in_flange).as_matrix()
    except ImportError:
        print("错误：需要安装 scipy 库来处理四元数。请运行 'pip install scipy'")
        sys.exit(1)
    except ValueError as e:
        print(f"错误：无效的四元数: {tool_quat_in_flange}. 错误信息: {e}")
        sys.exit(1)


    # 3. 构建 T_tool2flange 变换矩阵
    T_tool_to_flange = np.identity(4)
    T_tool_to_flange[:3, :3] = tool_rot_in_flange
    T_tool_to_flange[:3, 3] = tool_pos_in_flange

    print("\n  工具相对于法兰的变换矩阵 (T_tool2flange):")
    print(T_tool_to_flange)

    # 4. 计算 T_tool2base
    T_tool_to_base = T_flange_to_base @ T_tool_to_flange

    print("\n最终结果：工具相对于基座的变换矩阵 (T_tool2base):")
    print(T_tool_to_base)

    # 5. (可选) 提取最终的位置和姿态
    final_tool_pos = T_tool_to_base[:3, 3]
    final_tool_rot_matrix = T_tool_to_base[:3, :3]
    try:
        final_tool_quat = R.from_matrix(final_tool_rot_matrix).as_quat()
        final_tool_euler_xyz_deg = R.from_matrix(final_tool_rot_matrix).as_euler('xyz', degrees=True)

        print("\n  最终工具位置 (X, Y, Z) in mm:", final_tool_pos)
        print("  最终工具姿态 (四元数 xyzw):", final_tool_quat)
        print("  最终工具姿态 (欧拉角 XYZ, 度):", final_tool_euler_xyz_deg)
    except ValueError as e:
         print(f"\n  无法从最终旋转矩阵提取姿态: {e}")


    # --- 计算工具相对于激光跟踪仪的位姿 ---
    print("\n计算工具相对于激光跟踪仪的位姿 (T_tool2laser = T_base2laser * T_tool2base):")

    # 1. 定义基座相对于激光跟踪仪的位姿 (T_base2laser)
    base_pos_in_laser = np.array([3610.8319, 3300.7233, 13.6472]) # X, Y, Z in mm
    # Quaternion format: [Rx, Ry, Rz, W] -> [x, y, z, w] for scipy
    base_quat_in_laser = np.array([0.0014, -0.0055, 0.7873, -0.6166]) # x, y, z, w

    # 确保四元数归一化
    base_quat_in_laser /= np.linalg.norm(base_quat_in_laser)

    print(f"  基座相对于激光的位置 (X,Y,Z): {base_pos_in_laser}")
    print(f"  基座相对于激光的姿态 (四元数 xyzw): {base_quat_in_laser}")

    # 2. 从四元数创建旋转矩阵
    try:
        # R already imported
        base_rot_in_laser = R.from_quat(base_quat_in_laser).as_matrix()
    except ValueError as e:
        print(f"错误：无效的基座四元数: {base_quat_in_laser}. 错误信息: {e}")
        sys.exit(1)

    # 3. 构建 T_base2laser 变换矩阵
    T_base_to_laser = np.identity(4)
    T_base_to_laser[:3, :3] = base_rot_in_laser
    T_base_to_laser[:3, 3] = base_pos_in_laser

    print("\n  基座相对于激光的变换矩阵 (T_base2laser):")
    print(T_base_to_laser)

    # 4. 计算 T_tool2laser
    # T_tool2laser = T_base2laser @ T_tool2base
    T_tool_to_laser = T_base_to_laser @ T_tool_to_base

    print("\n最终结果：工具相对于激光的变换矩阵 (T_tool2laser):")
    print(T_tool_to_laser)

    # 5. (可选) 提取最终工具在激光坐标系下的位置和姿态
    final_tool_pos_laser = T_tool_to_laser[:3, 3]
    final_tool_rot_matrix_laser = T_tool_to_laser[:3, :3]
    try:
        final_tool_quat_laser = R.from_matrix(final_tool_rot_matrix_laser).as_quat()
        final_tool_euler_xyz_deg_laser = R.from_matrix(final_tool_rot_matrix_laser).as_euler('xyz', degrees=True)

        print("\n  最终工具在激光坐标系下的位置 (X, Y, Z) in mm:", final_tool_pos_laser)
        print("  最终工具在激光坐标系下的姿态 (四元数 xyzw):", final_tool_quat_laser)
        print("  最终工具在激光坐标系下的姿态 (欧拉角 XYZ, 度):", final_tool_euler_xyz_deg_laser)
    except ValueError as e:
         print(f"\n  无法从最终工具在激光下的旋转矩阵提取姿态: {e}")


except Exception as e:
    print(f"\n在计算过程中发生错误: {e}")
    print("请检查:")
    print("1. forward_kinematics.py 文件中的 RobotKinematics 类及其 forward_kinematics 方法是否正确实现。")
    print("2. 输入的关节角度数量是否正确（通常是6个）。")
