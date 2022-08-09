# import threading
import itertools
import threading
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

import get_relief_data as relief
# import downdoad_srtm_data as d_srtm
import get_time


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


# h_start = 1
# h_cnt = 1
v_start = 4
v_cnt = 1

scale_in_sec = 3

thread1 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 1, 36, v_start, v_cnt, total_counter,
                                                          progress_counter))
thread2 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 37, 36, v_start, v_cnt, total_counter,
                                                          progress_counter))
# thread3 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 37, 18, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread4 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 55, 18, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread5 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 33, 8, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread6 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 41, 8, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread7 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 49, 8, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread8 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 57, 8, v_start, v_cnt, total_counter,
#                                                           progress_counter))
# thread9 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 65, 8, v_start, v_cnt, total_counter,
#                                                           progress_counter))

start_time = get_time.get_current_time()
print(start_time)

# thread1.start()
# thread2.start()
# thread3.start()
# thread4.start()
# thread5.start()
# thread6.start()
# thread7.start()
# thread8.start()
# thread9.start()

# thread1.join()
# thread2.join()
# thread3.join()
# thread4.join()

if progress_counter.get() == total_counter.get() * 25 and progress_counter.get() > 0:
    end_time = get_time.get_current_time()
    print(end_time + '\n')
    get_time.get_time_interval(start_time, end_time)

# conda activate gdal_pycharm
# python C:/Users/tum/PycharmProjects/gdal_pycharm/main.py

# res_arr = np.full((1201, 1201), 0)
# res1 = relief.read_hgt_file('N54W070.hgt', 1201, res_arr)
# plt.imshow(res1, cmap='gist_earth')
# plt.show()
#
# print(res1[100][:1000])
#
# res2 = relief.read_hgt_file('N59W174.hgt', 1201, res_arr)
# plt.imshow(res2, cmap='gist_earth')
# plt.show()
#
# print(res2[100][:1000])

''' Tiff image check'''
path_to_file = 'C:/Users/tum/Programming/SRTM_data/Aster GDEM V3 data/data/ASTGTMV003_N73E058_dem.tif'
path_to_file2 = 'C:/Users/tum/Programming/SRTM_data/srtm_data/22_24/srtm_22_24.tif'

ds = gdal.Open(path_to_file2)
band = ds.GetRasterBand(1)
elevations = band.ReadAsArray()
# elevations = relief.round_2d_array_to_default_scale(elevations, 3)
#
print(elevations.shape)
#
relief.replace_nodata_value_in_array(elevations)
long_min, long_delta, dxdy, lat_max, dydx, lat_delta = ds.GetGeoTransform()
print(lat_max, long_min)
#
print(elevations[1000][:1000])
#
plt.imshow(elevations, cmap='gist_earth')
# # , extent=[long_min, long_min + 1, lat_max - 1, lat_max]
plt.show()



# grid = '38_03'
# path_to_file2 = 'C:/Users/tum/Programming/SRTM_data/srtm_data/' + grid + '/srtm_' + grid + '.tif'
# ds2 = gdal.Open(path_to_file2)
# band2 = ds2.GetRasterBand(1)
# elevations2 = band2.ReadAsArray()
#
# long_min, long_delta, dxdy, lat_max, dydx, lat_delta = ds2.GetGeoTransform()
#
# plt.title("srtm_plus15")
# plt.imshow(rounded_elevations, cmap='gist_earth')
# , extent=[long_min, long_min + 1, lat_max - 1, lat_max]
# plt.show()
#
# plt.title("just srtm")
# plt.imshow(elevations2, cmap='gist_earth', extent=[long_min, long_min + 5, lat_max - 5, lat_max])
# plt.show()

# thread1 = threading.Thread(target=d_srtm.download_srtm_data, args=[1, 72, 22])
# thread2 = threading.Thread(target=d_srtm.download_srtm_data, args=[1, 72, 23])
# thread3 = threading.Thread(target=d_srtm.download_srtm_data, args=[1, 72, 24])
# thread1.start()
# thread2.start()
# thread3.start()
