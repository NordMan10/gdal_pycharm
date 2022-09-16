import datetime
import os
import threading
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

import get_time
import filenames as fn
import dirs_difference as dirs_diff
import copy_files
import get_relief_data as relief
import get_tiles_from_aster as aster


class Counter:
    counter = 0

    def get(self):
        return self.counter

    def increment(self):
        self.counter += 1

    def decrement(self):
        self.counter -= 1

    def add(self, value):
        self.counter += value

    def subtract(self, value):
        self.counter -= value


total_counter = Counter()
progress_counter = Counter()


# copy_files.copy_files_from_multiple_folders(r'C:\Users\tum\Programming\SRTM_data\res\6x6',
#                                             r'C:\Users\tum\Programming\Map\maps\SRTM\6x6')

# thread1 = threading.Thread(target=copy_files.move_files_from_to, args=(r'\\idv.nita.ru\denis_buffer\Relief\6x6\N00-S01',
#                                                                        r'\\idv.nita.ru\denis_buffer\Relief\6x6'))
# thread2 = threading.Thread(target=copy_files.move_files_from_to, args=(r'\\idv.nita.ru\denis_buffer\Relief\6x6\N02-01',
#                                                                        r'\\idv.nita.ru\denis_buffer\Relief\6x6'))
# thread3 = threading.Thread(target=copy_files.move_files_from_to, args=(r'\\idv.nita.ru\denis_buffer\Relief\6x6\N04-03',
#                                                                        r'\\idv.nita.ru\denis_buffer\Relief\6x6'))
# thread4 = threading.Thread(target=copy_files.move_files_from_to, args=(r'\\idv.nita.ru\denis_buffer\Relief\6x6\N06-05',
#                                                                        r'\\idv.nita.ru\denis_buffer\Relief\6x6'))
# thread5 = threading.Thread(target=copy_files.move_files_from_to, args=(r'\\idv.nita.ru\denis_buffer\Relief\6x6\N08-07',
#                                                                        r'\\idv.nita.ru\denis_buffer\Relief\6x6'))


start_time = get_time.get_current_time()
print(start_time)

# thread1.start()
# thread2.start()
# thread3.start()
# thread4.start()
# thread5.start()
#
# thread1.join()
# thread2.join()
# thread3.join()
# thread4.join()
# thread5.join()

if progress_counter.get() == total_counter.get() and progress_counter.get() > 0:
    end_time = get_time.get_current_time()
    print(end_time + '\n')
    get_time.get_time_interval(start_time, end_time)


# path_to_aster_file = r'D:\ReliefProject\AsterGdemV3Data\data\N54-50\ASTGTMV003_N52E004_dem.tif'
# path_to_dest_dir = r'D:\ReliefProject\res\3x3\Astra\aster\N54-50'
# aster.create_one_hgt_file_from_aster_tif(path_to_aster_file, path_to_dest_dir)

# conda activate gdal_pycharm
# python C:/Users/tum/PycharmProjects/gdal_pycharm/main.py

fn.increase_lat(r'C:\Users\tum\Programming\SRTM_data\res\30x30\N89-80')

# dirs_diff.main(r'D:\ReliefProject\AsterGdemV3Data\data\N82-80', r'D:\ReliefProject\res\3x3\N82-80')

path_to_file_to_read = r'D:\ReliefProject\res\3x3\Astra\aster\N54-50\3x3\N50E000.hgt'

splitted_path =path_to_file_to_read.split('\\')

# scale_in_secs = int(splitted_path[len(splitted_path) - 2].split('x')[0])
scale_in_deg = 5

pole = splitted_path[len(splitted_path) - 1][: 1]
hemisphere = splitted_path[len(splitted_path) - 1][3: 4]
lat_min = int(splitted_path[len(splitted_path) - 1][1: 3])
lon_min = int(splitted_path[len(splitted_path) - 1][4: 7])
lat_max = lat_min + scale_in_deg if pole == 'N' else lat_min - scale_in_deg
lon_max = lon_min - scale_in_deg if hemisphere == 'W' else lon_min + scale_in_deg
#
# res = relief.read_hgt_file(path_to_file_to_read)
# plt.imshow(res, cmap='gist_earth', extent=[lon_min, lon_max, lat_min, lat_max])
# plt.title(splitted_path[len(splitted_path) - 1])
# plt.show()
start = datetime.datetime.now()

path_to_file_to_read2 = r'C:\Users\tum\Programming\Map\maps\SRTM\60x60\N50E000.hgt'
splitted_path = path_to_file_to_read2.split('\\')

# scale_in_secs = int(splitted_path[len(splitted_path) - 2].split('x')[0])
scale_in_deg = 1

pole = splitted_path[len(splitted_path) - 1][: 1]
hemisphere = splitted_path[len(splitted_path) - 1][3: 4]
lat_min = int(splitted_path[len(splitted_path) - 1][1: 3])
lon_min = int(splitted_path[len(splitted_path) - 1][4: 7])
lat_max = lat_min + scale_in_deg if pole == 'N' else lat_min - scale_in_deg
lon_max = lon_min - scale_in_deg if hemisphere == 'W' else lon_min + scale_in_deg

# res2 = relief.read_hgt_file(path_to_file_to_read2)
# plt.imshow(res2, cmap='gist_earth', extent=[lon_min, lon_max, lat_min, lat_max])
# plt.title(splitted_path[len(splitted_path) - 1])
# plt.show()
#
# end = datetime.datetime.now()
# print(end - start)


