#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取CSV数据文件中的测量值列（第8-13列）
跳过标题行，输出每一行的测量坐标和角度值
"""

import os
import csv
import sys

def extract_measurements(input_file, output_file):
    """
    从CSV文件中提取测量值列（列索引7-12，对应第8-13列）
    跳过第一行（标题行）
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in, delimiter=' ')
        writer = csv.writer(f_out, delimiter=',')
        
        # 添加标题行
        writer.writerow(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        
        # 跳过第一行（标题行）
        next(reader)
        
        # 处理每一行数据
        for row in reader:
            if not row:  # 跳过空行
                continue
           
            measurements = row[7:13]  # 第8到13列的数据
            
            # 写入输出文件
            writer.writerow(measurements)
    
    print(f"测量值数据已提取到: {output_file}")

if __name__ == "__main__":
    # 默认输入输出文件路径
    input_file = "data/data.csv"
    output_file = "data/tool_pos_laser.csv"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    extract_measurements(input_file, output_file)