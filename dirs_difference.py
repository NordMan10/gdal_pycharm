import os


def main(source_dir, dest_dir):
    dir1 = source_dir
    dir2 = dest_dir

    dir1_content_list = []
    dir2_content_list = os.listdir(dir2)

    filenames1 = os.listdir(dir1)

    filtered = [filename for filename in filenames1 if not filename[11: 18] + '.hgt' in dir2_content_list]
    [print(i + '\n') for i in filtered]


