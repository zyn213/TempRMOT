import os
import re
import json
print(os.getcwd())


def count_unique_video_names(input_file):
    """
    统计文件中不同 video_name 的个数
    """
    video_names = set()  # 用集合存储不同的 video_name（自动去重）

    # 打开文件并逐行读取
    with open(input_file, 'r') as f_in:
        for line in f_in:
            # 使用正则表达式提取 video_name（在 'OVIS/training/' 和 '/' 之间）
            match = re.search(r'OVIS/training/([^/]+)/', line)
            if match:
                video_name = match.group(1)  # 提取到的 video_name
                video_names.add(video_name)  # 将 video_name 添加到集合中

    return len(video_names)


def save_image_paths(training_set, training_dir, output_file):
    """
    Save image paths from the training_set into a file.

    Args:
    - training_set (list): List of folder names to look for.
    - training_dir (str): Path to the OVIS training directory.
    - output_file (str): Path to the output file.
    """
    paths_to_save = []
    for folder_name in training_set:
        folder_path = os.path.join(training_dir, folder_name)
        if os.path.isdir(folder_path):  # Check if the folder exists
            for file_name in sorted(os.listdir(folder_path)):  # Ensure sorted file order
                if file_name.endswith('.jpg'):  # Only process .jpg files
                    # Create relative path starting from "OVIS"
                    relative_path = os.path.relpath(os.path.join(folder_path, file_name),
                                                    start=training_dir.split("/OVIS")[0])
                    paths_to_save.append(relative_path)
        else:
            print(f"Warning: Folder {folder_name} does not exist in {training_dir}")

    # Write all collected paths to the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure output directory exists
    with open(output_file, 'w') as f:
        for path in paths_to_save:
            f.write(f"{path}\n")
    print(f"Saved {len(paths_to_save)} paths to {output_file}")



# Main function
if __name__ == "__main__":
    # for example: take 9 videos as training set
    training_set = ["001ca3cb", "00501424", "00e92ab4", "02deca50", "033333fd", "035359e5", "03c21af7", "03c99e83", "04ae7a6b"]
    training_dir = "/nfs/data3/shuaicong/refer-ovis/OVIS/training"
    output_file = "../datasets/data_path/refer-ovis-9.train"
    save_image_paths(training_set, training_dir, output_file)
    unique_video_count = count_unique_video_names(output_file)
    print(f"文件中共有 {unique_video_count} 个不同的 video_name。")

    # 读取 training.json 文件
    training_json_path = "tools/OVIS-training-videos-name.json"  # 替换为实际的 json 文件路径
    with open(training_json_path, 'r') as f:
        training_data = json.load(f)  # 读取为列表

    # 转换为集合（如果需要唯一性）
    training_set = set(training_data)
    output_file_full = "datasets/data_path/refer-ovis.train"
    save_image_paths(training_set, training_dir, output_file_full)



