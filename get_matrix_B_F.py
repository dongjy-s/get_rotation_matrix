import csv
from forward_kinematics import RobotKinematics

class MatrixConverterB2F:
    """
    用于计算机器人关节角度对应的变换矩阵的类
    """
    def __init__(self):
        # 创建机器人运动学实例
        self.robot = RobotKinematics()
    
    def joint_angles_to_matrices(self, angles_file_path):
        """
        读取关节角度文件，计算对应的位姿并转换为旋转矩阵
        
        参数:
        angles_file_path -- 关节角度CSV文件路径
        
        返回:
        transforms_list -- 包含所有4x4变换矩阵的列表（表示基坐标系到法兰坐标系的变换）
        """
        # 存储结果的列表
        transforms_list = []
        
        # 读取关节角度文件
        with open(angles_file_path, 'r') as file:
            reader = csv.reader(file)
            
            # 跳过第一行（标题行或注释行）
            next(reader, None)
            
            for row in reader:
                # 检查是否为注释行
                if not row or (len(row) > 0 and row[0].startswith('#')):
                    continue
                    
                # 将字符串转换为浮点数
                joint_angles = [float(val) for val in row]
                
                # 确保有6个关节角度
                if len(joint_angles) != 6:
                    continue
                
                try:
                    # 计算正运动学
                    _, _, T_flange = self.robot.forward_kinematics(joint_angles)
                    
                    # 存储结果
                    transforms_list.append(T_flange)
                    
                except ValueError:
                    pass
        
        return transforms_list
    
    def calculate_from_angles(self, joint_angles):
        """
        直接从关节角度计算变换矩阵
        
        参数:
        joint_angles -- 包含6个关节角度的列表
        
        返回:
        T_flange -- 4x4变换矩阵
        """
        try:
            # 计算正运动学
            _, _, T_flange = self.robot.forward_kinematics(joint_angles)
            return T_flange
        except ValueError as e:
            print(f"计算错误: {e}")
            return None

if __name__ == "__main__":
    # 创建计算器实例
    calculator = MatrixConverterB2F()
    
    # 输入文件路径
    angles_file_path = "data/formatted_angle.csv"
    
    # 计算变换矩阵
    transforms_list = calculator.joint_angles_to_matrices(angles_file_path)
    
    # 打印第一个变换矩阵作为参考
    if transforms_list:
        print("第一个关节角度对应的变换矩阵（基座表示法兰）:")
        print(transforms_list[0])