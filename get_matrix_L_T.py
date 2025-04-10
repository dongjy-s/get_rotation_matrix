import numpy as np
from scipy.spatial.transform import Rotation as R
import csv

class MatrixConverterL2T:
    """
    用于将位置和四元数转换为变换矩阵的类
    """
    @staticmethod
    def pos2matrix(pos):
        """
        将位置和四元数转换为4x4变换矩阵
        
        参数:
        pos -- 包含位置和四元数的数组 [x, y, z, qx, qy, qz, qw]
        
        返回:
        transform_matrix_L_T -- 4x4变换矩阵
        """
        # 提取位置和四元数
        x, y, z = pos[0], pos[1], pos[2]
        qx, qy, qz, qw = pos[3], pos[4], pos[5], pos[6]
        
        # 归一化四元数以确保其有效性
        quat = np.array([qx, qy, qz, qw])
        quat_norm = quat / np.linalg.norm(quat)
        
        # 使用scipy的Rotation模块创建旋转矩阵
        rotation = R.from_quat(quat_norm)
        rotation_matrix = rotation.as_matrix()
        
        # 创建4x4变换矩阵
        transform_matrix_L_T = np.eye(4)
        transform_matrix_L_T[:3, :3] = rotation_matrix
        transform_matrix_L_T[:3, 3] = [x, y, z]
        
        return transform_matrix_L_T

    def process_data_file(self, file_path):
        """
        处理CSV文件数据，将每行数据转换为变换矩阵
        
        参数:
        file_path -- CSV文件路径
        
        返回:
        matrices -- 包含所有变换矩阵的列表
        """
        matrices = []
        
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # 读取标题行
                
                if header is None:
                    print(f"警告: {file_path} 文件为空")
                    return matrices
                
                # 检查文件数据是否至少包含7列（x,y,z,qx,qy,qz,qw）
                if len(header) < 7:
                    print(f"警告: {file_path} 数据格式不正确，至少需要7列")
                    return matrices
                
                row_count = 0
                for row in reader:
                    try:
                        # 将字符串转换为浮点数
                        if len(row) < 7:
                            print(f"跳过行 {row_count+1}: 数据不完整")
                            continue
                            
                        data = [float(val) for val in row[:7]]  # 只取前7个值
                        
                        # 检查数据有效性
                        if np.isnan(data).any() or np.isinf(data).any():
                            print(f"跳过行 {row_count+1}: 数据包含NaN或Inf")
                            continue
                            
                        # 检查四元数部分是否全为0
                        if np.allclose(data[3:7], np.zeros(4)):
                            print(f"跳过行 {row_count+1}: 四元数全为0")
                            continue
                        
                        # 转换为矩阵
                        matrix = self.pos2matrix(data)
                        matrices.append(matrix)
                        row_count += 1
                    except ValueError as e:
                        print(f"处理行 {row_count+1} 时出错: {e}")
                        continue
                
                print(f"成功处理了 {len(matrices)}/{row_count} 行数据")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
        
        return matrices


if __name__ == "__main__":
    # 创建转换器实例
    converter = MatrixConverterL2T()
    
    # 示例：处理CSV文件
    file_path = "data/formatted_data.csv"
    transformation_matrices = converter.process_data_file(file_path)
    
    # 打印第一个矩阵作为示例
    if transformation_matrices:
        print("第一个点的变换矩阵(激光雷达表示工具):")
        print(transformation_matrices[0])
        print(f"总共处理了 {len(transformation_matrices)} 个点")