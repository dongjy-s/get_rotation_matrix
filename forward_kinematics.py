import numpy as np
from scipy.spatial.transform import Rotation as R

class RobotKinematics:
    """
    机器人运动学计算类
    """
    def __init__(self):
        # 关节限位：
        self.joint_limits = [
            [-100, 100],
            [-90, 100],
            [-100, 60],
            [-100, 100],
            [-90, 90],
            [-120, 120]
        ]
       # 改进DH参数: [alpha_i, a_i, d_i, theta_offset_i]
        self.modified_dh_params = [
            [0, 0, 487, 0],
            [-90, 85, 0, -90],
            [0, 640, 0, 0],
            [-90, 205, 720, 0],
            [90, 0, 0, 0],
            [-90, 0, 75, 180]
        ]
        # self.modified_dh_params = [
        #     [0, 0, 487.4009, -0.0063],
        #     [-90, 85.5599, 0, -90.2267],
        #     [0, 639.8143, 0, -0.4689],
        #     [-90, 205.288, 720.3035, 0.5631],
        #     [90, 0, 0, 0.0723],
        #     [-90, 0, 75.7785, 179.6983]
        # ]
        
    def modified_dh_matrix(self, alpha_deg, a, d, theta_deg):
        """
        计算改进DH变换矩阵
        
        参数:
            alpha_deg: alpha角度 (度)
            a: 连杆长度 (mm)
            d: 连杆偏移 (mm)
            theta_deg: theta角度 (度)
                
        返回:
            numpy.ndarray: 4x4变换矩阵
        """
        # 将角度从度转换为弧度
        alpha_rad = np.deg2rad(alpha_deg)
        theta_rad = np.deg2rad(theta_deg)

        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        cos_alpha = np.cos(alpha_rad)
        sin_alpha = np.sin(alpha_rad)

        # 改进DH变换矩阵公式:
        # A_i = Rot_z(theta_i) * Trans_z(d_i) * Rot_x(alpha_i) * Trans_x(a_i)
        A = np.array([
            [cos_theta, -sin_theta, 0, a],
            [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d],
            [sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, cos_alpha*d],
            [0, 0, 0, 1]
        ])
        return A
    
    def check_joint_limits(self, joint_angles):
        """
        检查关节角度是否在限位范围内
        :param joint_angles: 关节角度列表，单位为度
        :raises: ValueError 如果任何关节角度超出限位范围
        """
        for i in range(6):
            if not (self.joint_limits[i][0] <= joint_angles[i] <= self.joint_limits[i][1]):
                raise ValueError(f"关节{i+1}角度{joint_angles[i]}超出限位范围[{self.joint_limits[i][0]}, {self.joint_limits[i][1]}]")
    
    def forward_kinematics(self, joint_angles):
        """
        计算机器人的正运动学
        :param joint_angles: 关节角度列表，单位为度
        :return: 末端位置和四元数
        :raises: ValueError 如果关节角度超出限位范围
        """
        # 检查关节限位
        self.check_joint_limits(joint_angles)
        
        # 初始化总变换矩阵 T_0^6 为单位矩阵
        T_flange = np.identity(4)
        
        # 循环计算每个关节的变换矩阵并累乘
        for i in range(6):
            # 获取当前关节的参数
            alpha_i = self.modified_dh_params[i][0]
            a_i = self.modified_dh_params[i][1]
            d_i = self.modified_dh_params[i][2]
            theta_offset_i = self.modified_dh_params[i][3]
            
            # 获取当前关节的可变角度 q_i
            q_i = joint_angles[i]
            
            # 计算实际的 theta_i = q_i + theta_offset_i
            theta_i = q_i + theta_offset_i
            
            # 计算当前关节的变换矩阵 A_i
            A_i = self.modified_dh_matrix(alpha_i, a_i, d_i, theta_i)
            
            # 累积变换: T_flange = T_flange * A_i
            T_flange = np.dot(T_flange, A_i)
        position = T_flange[:3, 3]
        
        # 提取旋转矩阵并转换为四元数
        rotation_matrix = T_flange[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # 返回的四元数格式为[x, y, z, w]
        
        return position, quaternion, T_flange



def main():
    # 设置 NumPy 打印选项，禁用科学计数法
    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
    
    # 创建机器人运动学实例
    robot = RobotKinematics()
    
    # 示例：使用默认的DH参数中的关节角度
    joint_angles = [2.636115843805,29.5175455057437,6.10038158777613,34.7483282137072,-47.3762053517954,35.7916251381285]
    
    # 计算正运动学
    position, quaternion, T_flange = robot.forward_kinematics(joint_angles)
     
    print("末端法兰位姿:")
    print(T_flange)
    print("------------------------------")
    print("末端位置 (x, y, z):", position)
    print("------------------------------")
    print("末端姿态四元数 (x, y, z, w):", quaternion)

if __name__ == "__main__":
    main()