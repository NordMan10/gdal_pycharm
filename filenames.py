import os
import re

# Open a file
dest_dir = r'D:\ReliefProject\res\3x3\S69-60'
# path = r"D:\ReliefProject\res\3x3\N69-65"


def pad_number2(match):
    number = int(match.group(1))
    return format(number, "02d")


def pad_number3(match):
    number = int(match.group(1))
    return format(number, "03d")


def increase_lat(dest_dir):
    # for dir in os.listdir(dest_dir):
    # if int(dir[1: 3]) > 60:
    end_path = dest_dir
    filenames = os.listdir(end_path)
    for i in range(len(filenames) - 1, -1, -1):
        if os.path.isfile(os.path.join(end_path, filenames[i])):
            lat_value = filenames[i][1: 3]
            lat_value = int(lat_value)
            lat_value += 5
            lat_value = re.sub(r"^(\d+)", pad_number2, str(lat_value))
            os.rename(os.path.join(end_path, filenames[i]), os.path.join(end_path, filenames[i][0] + lat_value + filenames[i][3:]))


def decrease_lat(dest_dir):
    # for dir in os.listdir(dest_dir):
    # if int(dir[1: 3]) > 60:
    end_path = dest_dir
    filenames = os.listdir(end_path)
    for i in range(0, len(filenames), 1):
        if os.path.isfile(os.path.join(end_path, filenames[i])):
            lat_value = filenames[i][1: 3]
            lat_value = int(lat_value)
            lat_value -= 5
            lat_value = re.sub(r"^(\d+)", pad_number2, str(lat_value))
            os.rename(os.path.join(end_path, filenames[i]),
                      os.path.join(end_path, filenames[i][0] + lat_value + filenames[i][3:]))


def increase_lon(dest_dir):
    # for dir in os.listdir(dest_dir):
    #     if int(dir[1: 3]) > 60:
    #         end_path = dest_dir + '\\' + dir
    filenames = os.listdir(dest_dir)
    for i in range(len(filenames) - 1, -1, -1):
        if os.path.isfile(os.path.join(dest_dir, filenames[i])):
            if filenames[i][: 4] == 'N64E':
                lon_value = filenames[i][4: 7]
                lon_value = int(lon_value)
                lon_value += 1
                lon_value = re.sub(r"^(\d+)", pad_number3, str(lon_value))
                os.rename(os.path.join(dest_dir, filenames[i]), os.path.join(dest_dir, filenames[i][0: 4] + lon_value + filenames[i][-4:]))


def decrease_lon(dest_dir):
    # for dir in os.listdir(dest_dir):
    #     if int(dir[1: 3]) > 60:
    #         end_path = dest_dir + '\\' + dir
    filenames = os.listdir(dest_dir)
    for i in range(0, len(filenames), 1):
        if os.path.isfile(os.path.join(dest_dir, filenames[i])):
            if filenames[i][: 4] == 'N69E':
                lon_value = filenames[i][4: 7]
                lon_value = int(lon_value)
                # if lon_value > 4:
                lon_value -= 1
                lon_value = re.sub(r"^(\d+)", pad_number3, str(lon_value))
                os.rename(os.path.join(dest_dir, filenames[i]), os.path.join(dest_dir, filenames[i][0: 4] + lon_value + filenames[i][-4:]))