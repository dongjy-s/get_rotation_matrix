# 机器人旋转矩阵计算与标定工具

## 项目简介

本项目提供机器人运动学计算和标定功能，主要用于：
1. 计算机器人正运动学（从关节角度到末端位姿）
2. 解决AX=YB标定问题（确定工具坐标系和基座坐标系关系）
3. 处理激光跟踪仪测量数据与机器人位姿数据的转换

## 安装要求

- Python 3.6+
- 依赖库：
```
numpy
scipy
opencv-python
pandas
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 主要功能模块

1. `forward_kinematics.py` - 机器人正运动学计算
   - 基于改进DH参数计算末端位姿
   - 关节限位检查
   - 输出变换矩阵和四元数

2. `calibrate.py` - AX=YB标定求解
   - 使用OpenCV的calibrateRobotWorldHandEye方法
   - 支持TSAI和PARK两种算法
   - 结果自动保存到results目录

3. 测试脚本（test目录）
   - 提供示例数据和测试用例

## 使用示例

1. 计算正运动学：
```python
from forward_kinematics import RobotKinematics

robot = RobotKinematics()
joint_angles = [7.06, 18.89, 42.24, 17.43, -70.12, 53.55]  # 示例关节角度
position, quaternion, T_flange = robot.forward_kinematics(joint_angles)
```

2. 执行AX=YB标定：
```python
from calibrate import calibrate_AX_equals_YB

# 加载关节角度和工具位姿数据
joint_angles_list = [...]  # 多组关节角度
tool_pos_list = [...]       # 对应的工具位姿

# 计算变换矩阵
A_list = calculate_T_flange(joint_angles_list)
B_list = tool_pos_to_transform_matrix(tool_pos_list)

# 执行标定
X, Y = calibrate_AX_equals_YB(A_list, B_list)
```

## 数据格式

- 关节角度：6个值的列表/数组（单位：度）
- 工具位姿：[x,y,z,rx,ry,rz]（位置mm，欧拉角度度）
- 变换矩阵：4x4 numpy数组
