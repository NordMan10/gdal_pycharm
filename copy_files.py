import os
import shutil
#
# source_folder = r"E:\demos\files\reports\\"
# destination_folder = r"E:\demos\files\account\\"


def move_files_from_to(source_folder, destination_folder):
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + r'\\' + file_name
        destination = destination_folder + r'\\' + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.move(source, destination)
            print('Moved:', file_name)


def copy_files_from_to(source_folder, destination_folder):
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + r'\\' + file_name
        destination = destination_folder + r'\\' + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('Copied:', file_name)


def move_files_from_multiple_folders(source_dir, dest_dir):
    for dir_name in os.listdir(source_dir):
        if dir_name[7:] != '.hgt':
            move_files_from_to(source_dir + r'\\' + dir_name, dest_dir)


def copy_files_from_multiple_folders(source_dir, dest_dir):
    for dir_name in os.listdir(source_dir):
        if dir_name[7:] != '.hgt':
            copy_files_from_to(source_dir + r'\\' + dir_name, dest_dir)

