import threading
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

import get_time
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


total_counter = Counter()
progress_counter = Counter()


def get_relief_files(scale_in_sec, h_start, h_cnt, v_start, v_cnt, total_counter, progress_counter):
    res_dict = relief.main(scale_in_sec, h_start, h_cnt, v_start, v_cnt, total_counter)
    relief.write_hgt_file(res_dict, total_counter, progress_counter)


def get_relief_files_from_aster(pole, start_lon, end_lon, total_counter, progress_counter):
    res_dict = aster.get_tiles_by_coords_6x6(pole, 61, 60, start_lon, end_lon, total_counter)
    aster.write_hgt_file(pole, res_dict, total_counter, progress_counter)


# h_start = 1
h_cnt = 24
v_start = 1
v_cnt = 4

scale_in_sec = 6

# thread1 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 1, h_cnt, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread2 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 25, h_cnt, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread3 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 49, h_cnt, v_start, v_cnt, total_counter,
#                                                           progress_counter))


thread1 = threading.Thread(target=get_relief_files_from_aster, args=('N', 0, 60, total_counter, progress_counter))
thread2 = threading.Thread(target=get_relief_files_from_aster, args=('N', 61, 120, total_counter, progress_counter))
thread3 = threading.Thread(target=get_relief_files_from_aster, args=('N', 121, 180, total_counter, progress_counter))


start_time = get_time.get_current_time()
print(start_time)

thread1.start()
# thread2.start()
# thread3.start()

thread1.join()
# thread2.join()
# thread3.join()

if progress_counter.get() == total_counter.get() and progress_counter.get() > 0:
    end_time = get_time.get_current_time()
    print(end_time + '\n')
    get_time.get_time_interval(start_time, end_time)


# conda activate gdal_pycharm
# python C:/Users/tum/PycharmProjects/gdal_pycharm/main.py


# res_arr1 = np.full((1201, 1201), 0)
# res1 = relief.read_hgt_file('D:/ReliefProject/res/6x6/N58W075.hgt', 1201, res_arr1)
# plt.imshow(res1, cmap='gist_earth')
# plt.show()

# print(res1[100][:1000])

# res_arr2 = np.full((1201, 1201), 0)
# res2 = relief.read_hgt_file('D:/ReliefProject/res/N60W075.hgt', 1201, res_arr2)
# plt.imshow(res2, cmap='gist_earth')
# plt.show()
#
# print(res2[100][:1000])

''' Tiff image check'''
# path_to_file = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60W001_dem.tif'
# # path_to_file2 = 'C:/Users/tum/Programming/SRTM_data/srtm_data/22_24/srtm_22_24.tif'
# #
# ds = gdal.Open(path_to_file)
# band = ds.GetRasterBand(1)
# elevations = band.ReadAsArray()
# elevations = elevations[0: -1]
# print(np.shape(elevations))
# elevations = relief.round_2d_array_to_default_scale(elevations, 3)
# #
# print(elevations.shape)
# #
# # relief.replace_nodata_value_in_array(elevations)
# long_min, long_delta, dxdy, lat_max, dydx, lat_delta = ds.GetGeoTransform()
# print(lat_max, long_min)
# #
# print(elevations[1000][:1000])
# #
# plt.imshow(elevations, cmap='gist_earth', extent=[long_min, long_min + 1, lat_max - 1, lat_max])
# # # , extent=[long_min, long_min + 1, lat_max - 1, lat_max]
# plt.show()

''''''
# rects = []

# path_to_file1 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E004_dem.tif'
# ds1 = gdal.Open(path_to_file1)
# band1 = ds1.GetRasterBand(1)
# elevations1 = band1.ReadAsArray()
# rects.append(elevations1)
#
# path_to_file2 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E005_dem.tif'
# ds2 = gdal.Open(path_to_file2)
# band2 = ds2.GetRasterBand(1)
# elevations2 = band2.ReadAsArray()
# rects.append(elevations2)
#
# path_to_file3 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E006_dem.tif'
# ds3 = gdal.Open(path_to_file3)
# band3 = ds3.GetRasterBand(1)
# elevations3 = band3.ReadAsArray()
# rects.append(elevations3)
#
# path_to_file4 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E007_dem.tif'
# ds4 = gdal.Open(path_to_file4)
# band4 = ds4.GetRasterBand(1)
# elevations4 = band4.ReadAsArray()
# rects.append(elevations4)
#
# path_to_file5 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E004_dem.tif'
# ds5 = gdal.Open(path_to_file5)
# band5 = ds5.GetRasterBand(1)
# elevations5 = band5.ReadAsArray()
# rects.append(elevations5)
#
# path_to_file6 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E005_dem.tif'
# ds6 = gdal.Open(path_to_file6)
# band6 = ds6.GetRasterBand(1)
# elevations6 = band6.ReadAsArray()
# rects.append(elevations6)
#
# path_to_file7 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E006_dem.tif'
# ds7 = gdal.Open(path_to_file7)
# band7 = ds7.GetRasterBand(1)
# elevations7 = band7.ReadAsArray()
# rects.append(elevations7)
#
# path_to_file8 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E007_dem.tif'
# ds8 = gdal.Open(path_to_file8)
# band8 = ds8.GetRasterBand(1)
# elevations8 = band8.ReadAsArray()
# rects.append(elevations8)
#
# path_to_file9 = 'D:/ReliefProject/AsterGdemV3Data/data/N64-60/ASTGTMV003_N60E007_dem.tif'
# ds9 = gdal.Open(path_to_file9)
# band9 = ds9.GetRasterBand(1)
# elevations9 = band9.ReadAsArray()
# rects.append(elevations9)

# res = aster.combine_rects(np.asarray(rects))
# print(np.shape(res))

