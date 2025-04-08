import os

def count_files_in_directory(directory):
    """
    统计指定目录及其子目录中的文件总数。

    :param directory: 文件夹路径
    :return: 文件总数
    """
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在！")
        return 0

    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    
    return file_count

if __name__ == "__main__":
    # 替换为你的文件夹路径
    folder_path = "real_tac_data"

    # 统计文件数量
    total_files = count_files_in_directory(folder_path)
    print(f"{folder_path} 文件夹中包含的文件总数为: {total_files}")