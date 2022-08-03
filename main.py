# import threading
import itertools
import threading

import numpy as np

import get_relief_data
import get_relief_data as relief
# import downdoad_srtm_data as d_srtm


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
    get_relief_data.write_hgt_file(res_dict, total_counter, progress_counter)


h_start = 1
# h_cnt = 1
v_start = 2
v_cnt = 1

scale_in_sec = 3

thread1 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 1, 8, v_start, v_cnt, total_counter, progress_counter))
thread2 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 9, 8, v_start, v_cnt, total_counter, progress_counter))
thread3 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 17, 8, v_start, v_cnt, total_counter, progress_counter))
thread4 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 25, 8, v_start, v_cnt, total_counter, progress_counter))
thread5 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 33, 8, v_start, v_cnt, total_counter, progress_counter))
thread6 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 41, 8, v_start, v_cnt, total_counter, progress_counter))
thread7 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 49, 8, v_start, v_cnt, total_counter, progress_counter))
thread8 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 57, 8, v_start, v_cnt, total_counter, progress_counter))
thread9 = threading.Thread(target=get_relief_files, args=(scale_in_sec, 65, 8, v_start, v_cnt, total_counter, progress_counter))

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
thread8.start()
thread9.start()


# res_arr = np.full((1201, 1201), 0)
# get_relief_data.read_hgt_file('N59E084.hgt', 1201, res_arr)
# get_relief_data.read_hgt_file('N59E085.hgt', 1201, res_arr)


# path_to_file = 'C:/Users/tum/Programming/SRTM_data/srtm_data/srtm_plus15/srtm_N50E005(2).tif'
#
# ds = gdal.Open(path_to_file)
# band = ds.GetRasterBand(1)
# elevations = band.ReadAsArray()
#
# grid = '38_03'
# path_to_file2 = 'C:/Users/tum/Programming/SRTM_data/srtm_data/' + grid + '/srtm_' + grid + '.tif'
# ds2 = gdal.Open(path_to_file2)
# band2 = ds2.GetRasterBand(1)
# elevations2 = band2.ReadAsArray()
#
# long_min, long_delta, dxdy, lat_max, dydx, lat_delta = ds2.GetGeoTransform()
#
# plt.title("srtm_plus15")
# plt.imshow(elevations, cmap='gist_earth', extent=[5, 10, 45, 50])
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
