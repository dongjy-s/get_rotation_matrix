# 机器人正运动学计算与手眼标定

这个程序使用修正的DH参数法(MDH)计算机器人的正运动学，包括末端位置和姿态四元数。同时提供手眼标定功能，解决AX=YB问题，计算工具到法兰和激光传感器到基座的变换矩阵。

## 功能

- 基于MDH参数计算机器人各关节的变换矩阵
- 计算从基座到末端执行器的总变换矩阵
- 提取末端位置坐标(x, y, z)
- 将旋转矩阵转换为四元数表示(x, y, z, w)
- 检查关节角度是否在限位范围内
- 计算并输出末端法兰的完整位姿矩阵
- 使用OpenCV实现手眼标定，解决AX=YB问题
- 提供手动实现的Tsai手眼标定算法（无需依赖OpenCV）
- 计算工具到法兰(T_tool2flange)和激光传感器到基座(T_laser2base)的变换矩阵

## 使用方法

1. 确保安装了必要的Python库：
   ```
   pip install -r requirements.txt
   ```
   或手动安装：
   ```
   pip install numpy scipy opencv-python-headless
   ```
   注意：OpenCV仅在使用`hand_eye_calibration.py`时需要，使用手动实现的`hand_eye_calibration_manual.py`只需numpy和scipy。

2. 运行正运动学计算：
   ```
   python forward_kinematics.py
   ```

3. 运行手眼标定：
   - 使用OpenCV实现：
     ```
     python hand_eye_calibration.py
     ```
   - 使用手动实现（不依赖OpenCV）：
     ```
     python hand_eye_calibration_manual.py
     ```

## 数据文件说明

手眼标定需要以下数据文件：
- `data/joint_angle.csv`: 机器人关节角度数据，包含多组j1-j6角度值
- `data/tool_pos_laser.csv`: 工具相对于激光传感器的位姿数据，包含x,y,z,rx,ry,rz值

## 手眼标定原理

本项目实现了AX=YB问题的解法，其中：
- A: 法兰到基座的变换矩阵(T_flange2base)
- X: 工具到法兰的变换矩阵(T_tool2flange)，未知量
- Y: 激光传感器到基座的变换矩阵(T_laser2base)，未知量
- B: 工具到激光传感器的变换矩阵(T_tool2laser)

提供了两种实现方式：
1. 使用OpenCV的calibrateHandEye函数（hand_eye_calibration.py）
2. 手动实现的Tsai方法（hand_eye_calibration_manual.py）

## DH参数说明

程序中使用的DH参数如下：

| 关节 | Alpha (°) | a (mm) | d (mm) | theta (°) |
|------|----------|--------|--------|----------|
| 1    | 0        | 0      | 490    | 0        |
| 2    | -90      | 85     | 0      | -90      |
| 3    | 0        | 640    | 0      | 0        |
| 4    | -90      | 205    | 720    | 0        |
| 5    | 90       | 0      | 0      | 0        |
| 6    | -90      | 0      | 75     | 180      |

这些参数描述了机器人的几何结构，用于计算正运动学。