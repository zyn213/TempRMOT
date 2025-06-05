import os
import shutil


def find_unmatched_json_files(json_dir, folder_dir, output_txt):
    """
    查找在 json_dir 中存在但在 folder_dir 中不存在的 JSON 文件（仅对比名称，不含扩展名）。

    参数:
        json_dir (str): 包含 .json 文件的目录。
        folder_dir (str): 包含子文件夹的目录。
        output_txt (str): 输出未匹配文件名的 txt 文件名（默认保存在当前目录）。

    返回:
        List[str]: 未匹配的 JSON 文件名（不含扩展名）。
    """
    # 获取 .json 文件名（不含扩展名）
    json_files = [os.path.splitext(f)[0]
                  for f in os.listdir(json_dir)
                  if f.endswith('.json')]

    # 获取文件夹名
    folder_names = [name
                    for name in os.listdir(folder_dir)
                    if os.path.isdir(os.path.join(folder_dir, name))]

    # 找出未匹配的项
    unmatched = sorted(set(json_files) - set(folder_names))

    # 写入输出文件
    with open(output_txt, 'w') as f:
        for name in unmatched:
            f.write(name + '\n')
        print(f'saved in {output_txt}')

    return unmatched

def copy_matched_json_files(txt_path, source_dir, target_dir):
    # 读取 txt 中的每一行（去掉换行符）
    with open(txt_path, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    for name in names:
        json_filename = name + '.json'
        src_path = os.path.join(source_dir, json_filename)
        dst_path = os.path.join(target_dir, json_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            count += 1
        else:
            print(f"[未找到] 文件不存在: {src_path}")

    print(f"\n✅ 共复制 {count} 个 JSON 文件到: {target_dir}")

def check_gt_predict(folder_dir, output_txt):
    missing_folders = []

    # 遍历所有子文件夹
    for root, dirs, files in os.walk(folder_dir):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            gt_path = os.path.join(subfolder_path, 'gt.txt')
            predict_path = os.path.join(subfolder_path, 'predict.txt')

            # 检查是否缺少文件
            if not os.path.isfile(gt_path) or not os.path.isfile(predict_path):
                print(f"Missing files in: {subfolder_path}")
                missing_folders.append(subfolder_path)

    # 保存到文件中
    with open(output_txt, 'w') as f:
        for folder in missing_folders:
            f.write(folder + '\n')

    print(f"Done. Missing folder list saved to: {output_txt}")

# 示例调用
if __name__ == "__main__":

    print("当前工作目录是：", os.getcwd())
    json_dir = '/nfs/data3/shuaicong/TempRMOT/refer-mot20/expression/valid/MOT20-03-all'
    folder_dir = '/nfs/data3/shuaicong/RMOT/outputs/mot20/final1/results_epoch59/MOT20-03'
    output_txt = 'unmatched_json_names_mot20-03-1.txt'
    output_txt_missing_files = 'missing_gt_predict-mot20-03.txt'
    check_gt_predict(folder_dir, output_txt_missing_files)
    find_unmatched_json_files(json_dir, folder_dir, output_txt)

    json_dir = '/nfs/data3/shuaicong/TempRMOT/refer-mot20/expression/valid/MOT20-05-all'
    folder_dir = '/nfs/data3/shuaicong/RMOT/outputs/mot20/final1/results_epoch59/MOT20-05'
    output_txt = 'unmatched_json_names_mot20-05-1.txt'
    output_txt_missing_files = 'missing_gt_predict-mot20-05.txt'
    check_gt_predict(folder_dir, output_txt_missing_files)
    find_unmatched_json_files(json_dir, folder_dir, output_txt)

    txt_file = 'unmatched_json_names_mot20-03-1.txt'
    source_dir = '/nfs/data3/shuaicong/TempRMOT/refer-mot20/expression/valid/MOT20-03-all'
    target_dir = '/nfs/data3/shuaicong/TempRMOT/refer-mot20/expression/valid/MOT20-03'
    copy_matched_json_files(txt_file, source_dir, target_dir)

    txt_file = 'unmatched_json_names_mot20-05-1.txt'
    source_dir = '/nfs/data3/shuaicong/TempRMOT/refer-mot20/expression/valid/MOT20-05-all'
    target_dir = '/nfs/data3/shuaicong/TempRMOT/refer-mot20/expression/valid/MOT20-05'
    copy_matched_json_files(txt_file, source_dir, target_dir)




