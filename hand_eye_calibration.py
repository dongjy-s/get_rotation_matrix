import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import csv
import os
from get_matrix_L_T import MatrixConverterL2T
from get_matrix_B_F import MatrixConverterB2F

class HandEyeCalibration:
    def __init__(self):
        """
        初始化手眼标定类
        """

        
        self.matrixConverter = MatrixConverterL2T()
        self.robotCalculator = MatrixConverterB2F()
        
    def load_data(self, angle_file, data_file):
        """
        加载关节角度和激光跟踪仪数据文件
        
        参数:
        angle_file -- 关节角度CSV文件路径
        data_file -- 激光跟踪仪数据CSV文件路径
        
        返回:
        T_B_F_list -- 基座到法兰的变换矩阵列表
        T_L_T_list -- 激光跟踪仪到工具的变换矩阵列表
        """
        # 计算基座到法兰的变换矩阵列表
        T_B_F_list = self.robotCalculator.joint_angles_to_matrices(angle_file)
        
        # 计算激光跟踪仪到工具的变换矩阵列表
        T_L_T_list = self.matrixConverter.process_data_file(data_file)
        
        # 确保数据数量一致
        min_len = min(len(T_B_F_list), len(T_L_T_list))
        T_B_F_list = T_B_F_list[:min_len]
        T_L_T_list = T_L_T_list[:min_len]
        
        return T_B_F_list, T_L_T_list
    
    def transform_to_vector(self, rot_matrix, t):
        """
        将旋转矩阵和平移向量转换为参数向量
        
        参数:
        rot_matrix -- 3x3旋转矩阵
        t -- 3x1平移向量
        
        返回:
        params -- 包含旋转(四元数)和平移的参数向量
        """
        # 旋转矩阵转四元数 - 注意这里使用scipy.spatial.transform.Rotation
        # 而不是使用参数名R来引用旋转矩阵
        quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
        # 组合参数: [qx, qy, qz, qw, tx, ty, tz]
        return np.concatenate([quat, t])
    
    def vector_to_transform(self, params):
        """
        将参数向量转换为变换矩阵
        
        参数:
        params -- 包含旋转(四元数)和平移的参数向量 [qx, qy, qz, qw, tx, ty, tz]
        
        返回:
        T -- 4x4变换矩阵
        """
        quat = params[:4]  # [qx, qy, qz, qw]
        trans = params[4:7]  # [tx, ty, tz]
        
        # 创建旋转矩阵
        rot_matrix = R.from_quat(quat).as_matrix()
        
        # 创建变换矩阵
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = trans
        
        return T
    
    def cost_function(self, params, T_B_F_list, T_L_T_list):
        """
        优化的成本函数: ||T_B_F * T_F_T - T_B_L * T_L_T||
        
        参数:
        params -- 优化参数，包含T_B_L和T_F_T的参数 [T_B_L_params, T_F_T_params]
        T_B_F_list -- 基座到法兰的变换矩阵列表
        T_L_T_list -- 激光跟踪仪到工具的变换矩阵列表
        
        返回:
        error -- 所有姿态下的总误差
        """
        # 将参数分为两部分
        T_B_L_params = params[:7]  # T_B_L的7个参数 [qx, qy, qz, qw, tx, ty, tz]
        T_F_T_params = params[7:]  # T_F_T的7个参数 [qx, qy, qz, qw, tx, ty, tz]
        
        # 转换为变换矩阵
        T_B_L = self.vector_to_transform(T_B_L_params)
        T_F_T = self.vector_to_transform(T_F_T_params)
        
        # 计算所有姿态的误差
        total_error = 0
        for i in range(len(T_B_F_list)):
            # 手眼标定方程: T_B_F * T_F_T = T_B_L * T_L_T
            left_side = np.dot(T_B_F_list[i], T_F_T)
            right_side = np.dot(T_B_L, T_L_T_list[i])
            
            # 计算误差（矩阵范数）
            error = np.linalg.norm(left_side - right_side, 'fro')
            total_error += error
        
        return total_error
    
    def calibrate(self, T_B_F_list, T_L_T_list):
        """
        执行手眼标定，计算T_B_L和T_F_T
        
        参数:
        T_B_F_list -- 基座到法兰的变换矩阵列表
        T_L_T_list -- 激光跟踪仪到工具的变换矩阵列表
        
        返回:
        T_B_L -- 基座到激光跟踪仪的变换矩阵
        T_F_T -- 法兰到工具的变换矩阵
        """
        # 初始猜测值：单位旋转和零平移
        initial_T_B_L = np.eye(4)
        initial_T_F_T = np.eye(4)
        
        # 转换为参数向量
        initial_T_B_L_params = self.transform_to_vector(initial_T_B_L[:3, :3], initial_T_B_L[:3, 3])
        initial_T_F_T_params = self.transform_to_vector(initial_T_F_T[:3, :3], initial_T_F_T[:3, 3])
        
        # 组合初始参数
        initial_params = np.concatenate([initial_T_B_L_params, initial_T_F_T_params])
        
        # 进行优化
        result = minimize(
            self.cost_function,
            initial_params,
            args=(T_B_F_list, T_L_T_list),
            method='BFGS',
            options={'maxiter': 1000, 'disp': True}
        )
        
        # 提取优化结果
        optimized_params = result.x
        T_B_L_params = optimized_params[:7]
        T_F_T_params = optimized_params[7:]
        
        # 转换为变换矩阵
        T_B_L = self.vector_to_transform(T_B_L_params)
        T_F_T = self.vector_to_transform(T_F_T_params)
        
        return T_B_L, T_F_T, result.fun
    
    def evaluate_calibration(self, T_B_F_list, T_L_T_list, T_B_L, T_F_T):
        """
        评估标定结果的准确性
        
        参数:
        T_B_F_list -- 基座到法兰的变换矩阵列表
        T_L_T_list -- 激光跟踪仪到工具的变换矩阵列表
        T_B_L -- 计算得到的基座到激光跟踪仪的变换矩阵
        T_F_T -- 计算得到的法兰到工具的变换矩阵
        
        返回:
        mean_error -- 平均误差
        max_error -- 最大误差
        """
        errors = []
        
        for i in range(len(T_B_F_list)):
            # 手眼标定方程: T_B_F * T_F_T = T_B_L * T_L_T
            left_side = np.dot(T_B_F_list[i], T_F_T)
            right_side = np.dot(T_B_L, T_L_T_list[i])
            
            # 计算误差（矩阵范数）
            error = np.linalg.norm(left_side - right_side, 'fro')
            errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        return mean_error, max_error

def main():
    # 设置 NumPy 打印选项，禁用科学计数法
    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
    
    # 文件路径
    angle_file = "data/formatted_angle.csv"
    data_file = "data/formatted_data.csv"
    
    # 创建手眼标定实例
    calibrator = HandEyeCalibration()
    
    # 加载数据
    print("加载数据...")
    T_B_F_list, T_L_T_list = calibrator.load_data(angle_file, data_file)
    print(f"加载了 {len(T_B_F_list)} 组数据")
    
    # 执行标定
    print("执行手眼标定...")
    T_B_L, T_F_T, cost = calibrator.calibrate(T_B_F_list, T_L_T_list)
    
    # 评估结果
    mean_error, max_error = calibrator.evaluate_calibration(T_B_F_list, T_L_T_list, T_B_L, T_F_T)
    
    # 打印结果
    print("\n标定结果:")
    print("基座到激光跟踪仪的变换矩阵 (T_B_L):")
    print(T_B_L)
    print("\n法兰到工具的变换矩阵 (T_F_T):")
    print(T_F_T)
    print("\n标定误差:")
    print(f"优化成本: {cost}")
    print(f"平均误差: {mean_error}")
    print(f"最大误差: {max_error}")

if __name__ == "__main__":
    main()