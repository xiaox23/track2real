offset_list = [
    [2.3, -4.1, -3.2, 8.8, 1],
    [-3.0, 0.5, -6.3, 7.9, 1],
    [3.2, 0.5, 4.2, 3.4, 1],
    [-1.3, 2.1, 9.1, 7.9, 0],
    [-1.9, 0.3, -1.9, 5.4, 1],
    [-1.5, 1.0, -9.0, 4.9, 1],
    [1.8, -2.4, 5.0, 8.9, 1],
    [1.1, -2.7, 1.9, 3.6, 0],
    [-4.8, -4.0, 5.8, 6.3, 0],
    [-3.2, 1.1, 9.0, 8.8, 1],
    [1.2, 0.2, -3.2, 6.8, 0],
    [4.2, -4.9, -5.6, 4.8, 1],
    [4.8, 3.6, 4.9, 8.4, 0],
    [-4.2, -2.7, -3.3, 5.8, 0],
    [-4.0, 2.2, 3.7, 5.3, 1],
    [1.6, 0.3, -6.5, 5.7, 1],
    [3.6, -3.7, 0.2, 6.7, 1],
    [-3.0, 3.8, -3.8, 4.2, 0],
    [1.9, -2.4, 1.7, 3.1, 0],
    [1.4, -2.5, -6.1, 5.4, 1],
    [1.7, 1.5, 9.1, 8.2, 1],
    [-1.2, 0.1, 8.8, 8.6, 0],
    [-3.5, 3.5, 1.4, 4.5, 1],
    [4.9, -4.2, -3.9, 5.7, 1],
    [-4.2, 4.7, -4.8, 5.7, 0],
    [3.2, -2.3, 7.9, 5.8, 0],
    [-2.8, 0.8, 0.1, 7.2, 1],
    [0.8, -3.6, 0.5, 3.9, 0],
    [-3.6, 0.0, -0.9, 5.3, 0],
    [3.5, 4.7, -7.3, 3.4, 1],
    [-3.4, -1.4, 1.5, 4.5, 0],
    [-0.7, 4.6, 7.7, 7.9, 1],
    [4.9, -1.6, -3.2, 8.6, 1],
    [3.2, -0.1, 1.2, 5.1, 0],
    [3.3, 4.6, -5.5, 3.6, 1],
    [2.9, 4.1, 3.8, 4.2, 0],
    [0.1, 5.0, -9.2, 3.2, 0],
    [4.1, -3.9, 6.3, 7.9, 1],
    [-1.2, -3.8, 7.8, 7.8, 1],
    [0.7, -4.0, -3.2, 7.0, 0],
    [-1.2, -4.5, -2.7, 6.3, 1],
    [-4.5, 2.1, -2.5, 5.4, 1],
    [0.0, 2.3, 9.1, 4.0, 0],
    [0.2, 1.7, -6.4, 4.5, 1],
    [-2.7, -3.4, 7.5, 4.9, 1],
    [3.6, 0.3, 8.8, 8.6, 1],
    [-2.9, -0.6, 2.6, 6.5, 0],
    [-3.4, -1.9, 4.1, 6.6, 1],
    [2.3, 2.9, -8.5, 6.6, 1],
    [-3.9, 1.8, 9.9, 8.4, 1],
]

offset_0 = []
offset_1 = []

for aa in offset_list:
    if aa[-1] == 0:
        offset_0.append(aa[:-1])
    else:
        offset_1.append(aa[:-1])

new_offset_0 = offset_0[:len(offset_0)//2]
new_offset_1 = offset_1[:len(offset_1)//2]
print(new_offset_0)
print("length of new_offset_0: ", len(new_offset_0))
print(new_offset_1)
print("length of new_offset_1: ", len(new_offset_1))

import random

def generate_data(n=100):
    """生成n组随机数据"""
    data = []
    for _ in range(n):
        x = round(random.uniform(-5,5),1  )# x 范围 [0, 100]
        y = round(random.uniform(-5,5),1 ) # y 范围 [0, 100]
        theta = round(random.uniform(-10,10),1)  # theta 范围 [0, 360]
        z = round(random.uniform(-3,3)+6,1)  # z 范围 [0, 1]
        data.append([x, y, theta, z])
    return data

def check_duplicate_lists(lst):
    # 用于存储已经出现过的子列表（转换为元组）
    seen = set()
    for sub_list in lst:
        # 将子列表转换为元组
        sub_tuple = tuple(sub_list)
        if sub_tuple in seen:
            return True
        # 将元组添加到集合中
        seen.add(sub_tuple)
    return False

new_offset_0.extend(generate_data(25-len(new_offset_0)))
new_offset_1.extend(generate_data(25-len(new_offset_1)))

print(new_offset_0)
print("length of new_offset_0: ", len(new_offset_0))
print(new_offset_1)
print("length of new_offset_1: ", len(new_offset_1))

result_0 = check_duplicate_lists(new_offset_0)
print(result_0)
result_1 = check_duplicate_lists(new_offset_1)
print(result_1)

new_offset_0_all = new_offset_0 *3
new_offset_1_all = new_offset_1 *3
print(new_offset_0_all)
print("length of new_offset_0_all: ", len(new_offset_0_all))
print(new_offset_1_all)
print("length of new_offset_1_all: ", len(new_offset_1_all))





