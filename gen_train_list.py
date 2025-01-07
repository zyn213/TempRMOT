import os
import re

def generate_file_list(source_dir, output_file):
    """
    Generate a file listing all image files under directories starting with '0' in the source_dir.
    Save the results with paths starting from 'OVIS/' to output_file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open the output file for writing
    with open(output_file, 'w') as f_out:
        # Walk through the source directory
        for root, dirs, files in os.walk(source_dir):
            # Extract the relative path from 'OVIS'
            relative_root = os.path.relpath(root, start=os.path.dirname(source_dir))

            # Extract the video_name (last part of the path)
            video_name = os.path.basename(root)

            # Skip directories that don't start with '0'
            if not video_name.startswith('0'):
                continue

            # Collect and sort all image files in the current directory
            image_files = sorted([file for file in files if file.endswith('.jpg')])

            # Write the relative paths of the image files to the output file
            for image_file in image_files:
                # Construct the full relative path
                relative_path = os.path.join(relative_root, image_file)
                f_out.write(relative_path + '\n')

    print(f"File list generated at: {output_file}")

def add_prefix_to_lines(input_file, prefix="OVIS/"):
    """
    Reads the input file, adds the specified prefix to each line, and writes the modified lines to the output file.
    """
    with open(input_file, 'r') as f_in:
        # Read all lines from the input file
        lines = f_in.readlines()

    # Add prefix to each line
    lines_with_prefix = [prefix + line.strip() + '\n' for line in lines]

    # Write the modified lines to the output file
    with open(input_file, 'w') as f_out:
        f_out.writelines(lines_with_prefix)

    print(f"File with prefixes written to: {input_file}")

def filter_lines(input_file, unwanted_strings):
    """
    读取输入文件，检查每行是否包含任意一个目标字符串，
    如果包含则删除该行，否者将该行保留并覆盖写入文件。
    """
    with open(input_file, 'r') as f_in:
        # 读取所有行
        lines = f_in.readlines()

    # 打开文件准备覆盖写入
    with open(input_file, 'w') as f_out:
        for line in lines:
            # 检查当前行是否包含任意一个不需要的字符串
            if any(unwanted_str in line for unwanted_str in unwanted_strings):
                continue  # 如果包含目标字符串则跳过该行
            f_out.write(line)  # 否则写入文件

    print(f"文件 {input_file} 更新完成！")

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

# Main function
if __name__ == "__main__":
    # Define source directory and output file
    source_dir = "/nfs/data3/shuaicong/refer-ovis/OVIS/training"
    output_file = "datasets/data_path/refer-ovis.train"

    # Generate the file list
    generate_file_list(source_dir, output_file)
    add_prefix_to_lines(output_file)

    unwanted_strings = [
        "08746283", "0b02886a", "0b27d5f7",
        "0b4b662c", "0b5e5d29", "0b787263", "0b99f522", "0c13e1e7",
        "0d1fe009", "0d876a65", "0d967fd1", "0e4aeed9", "0e7001dd",
        "0f685e01"
    ]

    filter_lines(output_file, unwanted_strings)

    unique_video_count = count_unique_video_names(output_file)
    print(f"文件中共有 {unique_video_count} 个不同的 video_name。")




