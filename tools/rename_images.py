import os
import re

def rename_images_in_directory(root_dir):
    """
    Traverse the directory tree starting from root_dir,
    and rename all image files from img_0000001.jpg to 000001.jpg format.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("img_") and filename.endswith(".jpg"):
                # Extract the numeric part from the original filename
                original_number = filename[4:-4]  # Strip "img_" and ".jpg"
                if len(original_number) == 7 and original_number.isdigit():
                    # Convert the numeric part to 6 digits
                    new_number = f"{int(original_number):06d}"  # Keep leading zeros
                    new_filename = f"{new_number}.jpg"
                    # Construct full paths
                    old_path = os.path.join(dirpath, filename)
                    new_path = os.path.join(dirpath, new_filename)
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")

# Specify the root directory containing the subdirectories and images
root_directory = '/nfs/data3/shuaicong/refer-ovis/OVIS/training' # Replace your path
root_directory_valid = '/nfs/data3/shuaicong/refer-ovis/OVIS/valid'
rename_images_in_directory(root_directory)
rename_images_in_directory(root_directory_valid)
