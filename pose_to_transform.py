import numpy as np
# 禁用科学计数法
np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
from scipy.spatial.transform import Rotation as R
import csv

def euler_to_transform(position, euler_angles_deg):
    """
    将位置和欧拉角（XYZ外旋顺序，单位为度）转换为齐次变换矩阵
    
    参数:
        position: 末端位置 [x, y, z] (单位: mm)
        euler_angles_deg: 欧拉角 [rx, ry, rz] (单位: 度，XYZ外旋顺序)
    
    返回:
        4x4 齐次变换矩阵
    """
    # 将欧拉角从度转换为弧度
    euler_angles_rad = np.deg2rad(euler_angles_deg)
    
    # 创建旋转对象（XYZ外旋顺序）
    rotation = R.from_euler('XYZ', euler_angles_rad, degrees=False)
    
    # 构造齐次变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()  # 旋转部分
    transform[:3, 3] = position               # 平移部分
    
    return transform

def load_transforms(input_file):
    """
    从CSV文件读取位置和欧拉角，转换为变换矩阵并存储在列表中
    
    参数:
        input_file: 输入CSV文件路径（包含x,y,z,rx,ry,rz）
    
    返回:
        list: 包含所有变换矩阵的列表
    """
    transform_matrices = []
    
    with open(input_file, 'r') as f_in:
        reader = csv.DictReader(f_in)
        
        # 处理每一行数据
        for row in reader:
            # 提取位置和欧拉角
            position = [float(row['x']), float(row['y']), float(row['z'])]
            euler_angles = [float(row['rx']), float(row['ry']), float(row['rz'])]
            
            # 转换为变换矩阵并添加到列表
            transform = euler_to_transform(position, euler_angles)
            transform_matrices.append(transform)
    
    return transform_matrices

if __name__ == "__main__":
    # 示例用法
    input_file = "data/tool_pos_laser.csv"
    transforms = load_transforms(input_file)
    
    # 打印第一个变换矩阵作为示例
    print("第一个变换矩阵:")
    print(transforms[0]) 