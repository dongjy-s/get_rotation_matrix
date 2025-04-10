# 把excel表格中复制的数据格式化为csv文件
# 输入文件：data/data.csv
# 输出文件：data/formatted_data.csv
import csv
import numpy as np
from scipy.spatial.transform import Rotation

def euler_to_quaternion(rx, ry, rz):
    """
    将欧拉角(度)转换为四元数
    :param rx: 绕X轴旋转角度(度)
    :param ry: 绕Y轴旋转角度(度)
    :param rz: 绕Z轴旋转角度(度)
    :return: 四元数(x,y,z,w)
    """
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    quat = rotation.as_quat()
    return [quat[0], quat[1], quat[2], quat[3]]

def format_data(input_file, output_file):
    """
    格式化数据文件
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile)
        
        # 跳过标题行
        headers = next(reader)
        writer.writerow(['X', 'Y', 'Z', 'Quaternion_x', 'Quaternion_y', 'Quaternion_z', 'Quaternion_w'])
        
        for row in reader:
            # 提取位置和欧拉角
            x, y, z, rx, ry, rz = map(float, row)
            
            # 转换为四元数
            quat = euler_to_quaternion(rx, ry, rz)
            
            # 写入格式化数据
            writer.writerow([x, y, z, *quat])

if __name__ == "__main__":
    input_csv = "data/data.csv"
    output_csv = "data/formatted_data.csv"
    format_data(input_csv, output_csv)
    print(f"数据已格式化并保存到 {output_csv}")