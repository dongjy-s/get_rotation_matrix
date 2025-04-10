import re

def extract_angles(file_path):
    """提取关节角度数据并保存为CSV格式"""
    pattern = re.compile(r'j\d+ = j:\{([^}]+)\}')
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # 提取前6个数字，去除空格和最后的0
                angles = [float(x.strip()) for x in match.group(1).split(',')[:6]]
                results.append(angles)
    
    # 格式化为j1: [角度列表] 形式
    # 将结果保存为CSV文件
    with open('data/formatted_angle.csv', 'w') as f:
        # 添加列标题作为注释
        f.write('# j1,j2,j3,j4,j5,j6 \n')
        for angles in results:
            f.write(','.join(map(str, angles)) + '\n')
    return results

if __name__ == "__main__":
    angles = extract_angles('data/angle.csv')  # 修改为相对路径
    print("success!")