import os


def main():
    dir1 = r'D:\ReliefProject\AsterGdemV3Data\data\S83-80'
    dir2 = r'D:\ReliefProject\res\3x3\S83-80'

    dir1_content_list = []
    dir2_content_list = os.listdir(dir2)

    filenames1 = os.listdir(dir1)

    filtered = [filename for filename in filenames1 if not filename[11: 18] + '.hgt' in dir2_content_list]
    [print(i + '\n') for i in filtered]


