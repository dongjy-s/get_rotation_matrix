import numpy as np
# 禁用科学计数法
np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
import csv
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("警告：OpenCV 未安装，AX=YB 标定功能将不可用。")
from forward_kinematics import RobotKinematics
from pose_to_transform import  load_transforms

def load_joint_angles(input_file):
    """
    从CSV文件读取关节角度
    
    参数:
        input_file: 输入CSV文件路径（包含j1,j2,j3,j4,j5,j6）
    
    返回:
        list: 包含所有关节角度组的列表
    """
    joint_angles_list = []
    
    with open(input_file, 'r') as f_in:
        reader = csv.reader(f_in)
        next(reader) # 跳过标题行
        
        # 处理每一行数据
        for row in reader:
            # 提取前6个值并转换为浮点数
            try:
                angles = [float(val) for val in row[:6]]
                joint_angles_list.append(angles)
            except (ValueError, IndexError) as e:
                print(f"警告: 读取或转换行数据时出错: {row}, 错误: {e}. 跳过此行.")
                continue
    
    return joint_angles_list

def get_fk_transforms(joint_angles_list):
    """
    使用正运动学计算变换矩阵
    
    参数:
        joint_angles_list: 关节角度列表
    
    返回:
        list: 包含所有变换矩阵的列表
    """
    robot = RobotKinematics()
    transform_matrices = []
    
    for idx, angles in enumerate(joint_angles_list):
        try:
            _, _, T_flange = robot.forward_kinematics(angles)
            transform_matrices.append(T_flange)
        except Exception as e:
            print(f"警告: 计算第 {idx} 组角度的正运动学时出错: {angles}, 错误: {e}. 跳过此组数据.")
            continue
    
    return transform_matrices


def calibrate_ax_yb(T_flange2base_list, T_tool2laser_list, method=cv2.CALIB_HAND_EYE_TSAI):
    """
    使用 OpenCV 解决 AX = YB 问题。
    求解 T_tool2flange (X) 和 T_laser2base (Y)。

    参数:
        T_flange2base_list: 法兰到基座的变换矩阵列表 (A_i)
        T_tool2laser_list: 工具到激光传感器的变换矩阵列表 (B_i)
        method: OpenCV calibrateHandEye 使用的标定方法

    返回:
        (T_X, T_Y): T_tool2flange, T_laser2base 元组。若失败则返回 (None, None)。
    """
    if not OPENCV_AVAILABLE:
        print("错误: 需要 OpenCV 来执行 AX=YB 标定。")
        return None, None   

    n = len(T_flange2base_list)
    if n != len(T_tool2laser_list):
        print("错误: 输入的变换矩阵列表长度不一致。")
        return None, None
    if n < 3:
        print(f"错误: 需要至少3组位姿数据，当前只有 {n} 组。")
        return None, None

    R_A_prime_list = []  
    t_A_prime_list = []  
    R_B_prime_list = []
    t_B_prime_list = []

    print(f"计算 {n-1} 组相对运动...")
    # *计算相对运动 A'_i = inv(A_i) * A_{i+1} 和 B'_i = inv(B_i) * B_{i+1}
    for i in range(n - 1):
        try:
            A_i = T_flange2base_list[i]
            A_i_plus_1 = T_flange2base_list[i+1]
            A_i_inv = np.linalg.inv(A_i)
            A_prime = A_i_inv @ A_i_plus_1

            B_i = T_tool2laser_list[i]
            B_i_plus_1 = T_tool2laser_list[i+1]
            B_i_inv = np.linalg.inv(B_i)
            B_prime = B_i_inv @ B_i_plus_1

            R_A_prime_list.append(A_prime[:3, :3])
            t_A_prime_list.append(A_prime[:3, 3])
            R_B_prime_list.append(B_prime[:3, :3])
            t_B_prime_list.append(B_prime[:3, 3])

        except np.linalg.LinAlgError as e:
            print(f"警告: 计算第 {i} 组相对运动时矩阵求逆失败 ({e})，跳过此组。")
            continue # 跳过这一对相对运动

    if len(R_A_prime_list) < 2:
        print(f"错误: 计算得到的有效相对运动组数 ({len(R_A_prime_list)}) 不足 2，无法标定。")
        return None, None

    print(f"使用 {len(R_A_prime_list)} 组有效相对运动进行 calibrateHandEye 求解 Y (T_laser2base)...")
    # 调用 calibrateHandEye 求解 A' Y = Y B' 中的 Y
    # 输入: A' (flange relative) -> gripper2base in OpenCV
    # 输入: B' (tool/laser relative) -> target2cam in OpenCV
    # 输出: Y (laser2base) -> cam2gripper in OpenCV
    try:
        R_Y, t_Y = cv2.calibrateHandEye(
            R_gripper2base=R_A_prime_list,
            t_gripper2base=t_A_prime_list,
            R_target2cam=R_B_prime_list,
            t_target2cam=t_B_prime_list,
            method=method
        )
        T_Y = np.eye(4)
        T_Y[:3, :3] = R_Y
        T_Y[:3, 3] = t_Y.flatten()
        print("成功求解 T_Y (T_laser2base)。")
    except cv2.error as e:
        print(f"错误: OpenCV calibrateHandEye 失败: {e}")
        return None, None
    except Exception as e:
         print(f"错误: 调用 calibrateHandEye 时发生未知错误: {e}")
         return None, None


    # 使用 Y 和第一组数据求解 X: X = inv(A_0) @ Y @ B_0
    print("使用 T_Y 和第一组位姿数据求解 T_X (T_tool2flange)...")
    try:
        A_0 = T_flange2base_list[0]
        B_0 = T_tool2laser_list[0]
        A_0_inv = np.linalg.inv(A_0)

        T_X = A_0_inv @ T_Y @ B_0
        print("成功求解 T_X (T_tool2flange)。")
    except np.linalg.LinAlgError as e:
        print(f"错误: 计算 T_X 时矩阵求逆失败: {e}")
        return None, T_Y # 至少返回已求解的 Y
    except IndexError:
        print("错误: 无法访问 T_flange2base_list[0] 或 T_tool2laser_list[0] 来计算 T_X。")
        return None, T_Y # 至少返回已求解的 Y
    except Exception as e:
        print(f"错误: 计算 T_X 时发生未知错误: {e}")
        return None, T_Y


    return T_X, T_Y

if __name__ == "__main__":
    #! 1. 加载机器人法兰位姿数据 
    joint_angle_file = "data/joint_angle.csv"
    joint_angles = load_joint_angles(joint_angle_file)
    print(f"读取到 {len(joint_angles)} 组关节角度。")

    if joint_angles:
        print("通过正运动学计算法兰位姿 (T_flange2base)...")
        T_flange2base = get_fk_transforms(joint_angles)
        print(f"计算得到 {len(T_flange2base)} 个法兰变换矩阵。")
        if T_flange2base:
            print("第一个法兰变换矩阵 T_flange2base[0]:")
            print(T_flange2base[0])
    else:
        T_flange2base = []

    #!  2. 加载工具相对于激光传感器的位姿数据
    tool_pos_laser_file = "data/tool_pos_laser.csv"
    T_tool2laser = load_transforms(tool_pos_laser_file)
    print(f"读取到 {len(T_tool2laser)} 个工具到激光的变换矩阵。")
    if T_tool2laser:
        print("第一个工具到激光的变换矩阵 T_tool2laser[0]:")
        print(T_tool2laser[0])

    #! 3. 执行 AX = YB 标定
    if not T_flange2base or not T_tool2laser:
        print("\n错误：缺少法兰位姿或工具激光位姿数据，无法进行标定。")
    elif len(T_flange2base) != len(T_tool2laser):
        print(f"\n错误：法兰位姿 ({len(T_flange2base)}) 和工具激光位姿 ({len(T_tool2laser)}) 数量不匹配，无法标定。")
    else:
        print("\n开始执行 AX = YB 标定...")
        # 选择标定方法，例如 TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS
        calibration_method = cv2.CALIB_HAND_EYE_TSAI
        # calibration_method = cv2.CALIB_HAND_EYE_PARK
        # calibration_method = cv2.CALIB_HAND_EYE_HORAUD
        # calibration_method = cv2.CALIB_HAND_EYE_ANDREFF # Usually requires more poses
        # calibration_method = cv2.CALIB_HAND_EYE_DANIILIDIS # Usually requires more poses
        print(f"使用 OpenCV 方法: {calibration_method}")

        T_tool2flange, T_laser2base = calibrate_ax_yb(T_flange2base, T_tool2laser, method=calibration_method)

        if T_tool2flange is not None:
            print("\n标定结果 T_X (工具到法兰的变换矩阵 T_tool2flange):")
            print(T_tool2flange)

        if T_laser2base is not None:
            print("\n标定结果 T_Y (激光传感器到基座的变换矩阵 T_laser2base):")
            print(T_laser2base)

            # 计算并打印 T_base2laser
            try:
                T_base2laser = np.linalg.inv(T_laser2base)
                print("\n计算得到的基座到激光传感器的变换矩阵 (T_base2laser = inv(T_laser2base)):")
                print(T_base2laser)
            except np.linalg.LinAlgError:
                print("\n计算 T_laser2base 的逆矩阵失败。")
        else:
             # 如果 T_laser2base 是 None 但 T_tool2flange 不是 (虽然不太可能在此实现中发生)
             if T_tool2flange is not None:
                 print("\n标定部分成功，但未能求解 T_laser2base。")
             else:
                 print("\n标定失败。") 