import gdal
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import re

import progress_bar as pb


def pad_number2(match):
    number = int(match.group(1))
    return format(number, "02d")


def pad_number3(match):
    number = int(match.group(1))
    return format(number, "03d")


# Functions that return rectangles near to a current rectangle

def get_right_top_rect(current_col, current_row, srtm_tile_h_cnt, tile_arrays_by_grid_coords, cap_array):
    right_top_rect_coords = ""
    if current_col == srtm_tile_h_cnt:
        right_top_rect_coords = re.sub(r"^(\d+)", pad_number2, str(1)) + "_" + re.sub(r"^(\d+)", pad_number2,
                                                                                      str(current_row))
    else:
        right_top_rect_coords = re.sub(r"^(\d+)", pad_number2, str(current_col + 1)) + "_" + re.sub(r"^(\d+)",
                                                                                                    pad_number2,
                                                                                                    str(current_row))

    if right_top_rect_coords in tile_arrays_by_grid_coords:
        return tile_arrays_by_grid_coords[right_top_rect_coords]
    else:
        return cap_array


def get_left_bottom_rect(current_col, current_row, srtm_tile_v_cnt, tile_arrays_by_grid_coords, cap_array):
    left_bottom_rect_coords = ""
    if current_row == srtm_tile_v_cnt:
        return cap_array
    else:
        left_bottom_rect_coords = re.sub(r"^(\d+)", pad_number2, str(current_col)) + "_" + re.sub(r"^(\d+)",
                                                                                                  pad_number2,
                                                                                                  str(current_row + 1))

    if left_bottom_rect_coords in tile_arrays_by_grid_coords:
        return tile_arrays_by_grid_coords[left_bottom_rect_coords]
    else:
        return cap_array


def get_right_bottom_rect(current_col, current_row, srtm_tile_v_cnt, srtm_tile_h_cnt,
                          tile_arrays_by_grid_coords, cap_array):
    if current_row == srtm_tile_v_cnt:
        return cap_array

    right_bottom_rect_coords = ""
    if current_col == srtm_tile_h_cnt:
        right_bottom_rect_coords = re.sub(r"^(\d+)", pad_number2, str(1)) + "_" + re.sub(r"^(\d+)",
                                                                                         pad_number2,
                                                                                         str(current_row + 1))
    else:
        right_bottom_rect_coords = re.sub(r"^(\d+)", pad_number2, str(current_col + 1)) + "_" + re.sub(r"^(\d+)",
                                                                                                       pad_number2,
                                                                                                       str(current_row + 1))

    if right_bottom_rect_coords in tile_arrays_by_grid_coords:
        return tile_arrays_by_grid_coords[right_bottom_rect_coords]
    else:
        return cap_array


def get_scale_in_degrees_and_rect_side_in_px(seconds_cnt):
    min_scale = 3
    default_pixel_num = 1200

    scale_in_degrees = seconds_cnt / min_scale
    rect_side = default_pixel_num * scale_in_degrees

    return scale_in_degrees, rect_side


# The function takes big 2D array and returns smaller 2D array in according to passed sizes
# If the function can't satisfy passed sizes of array, it returns array with less sizes
def get_2d_array_from_bigger_2d_array(init_arr, left_border, top_border, tile_size, overlap):
    bottom_border = len(init_arr)
    if bottom_border - top_border > tile_size + overlap:
        bottom_border = top_border + tile_size + overlap
    right_border = len(init_arr)
    if right_border - left_border > tile_size + overlap:
        right_border = left_border + tile_size + overlap

    res_arr = []
    for i in range(bottom_border - top_border):
        res_arr += [init_arr[i + top_border, left_border: right_border]]

    return res_arr


def get_2d_index_by_element_number(ncols, number):
    element_row = 0

    while number > ncols:
        number -= ncols
        element_row += 1

    element_col = number - 1

    return element_row, element_col


def split_rect_in_defined_tiles(rects, tile_size, left_border, top_border, overlap):
    # Combine all rects to one entire rect
    two_top_combined_rects = np.hstack((rects[0], rects[1]))
    two_bottom_combined_rects = np.hstack((rects[2], rects[3]))
    entire_combined_rect = np.vstack((two_top_combined_rects, two_bottom_combined_rects))

    # Getting size of default rect (6000 px)
    rect_side_size = rects[0].shape[0]

    free_h_space = rect_side_size - left_border
    free_v_space = rect_side_size - top_border

    tiles_cnt_in_h_axis = math.ceil(free_h_space / tile_size)
    tiles_cnt_in_v_axis = math.ceil(free_v_space / tile_size)

    right_shift = tiles_cnt_in_h_axis * tile_size - free_h_space
    bottom_shift = tiles_cnt_in_v_axis * tile_size - free_v_space

    res_tiles = []

    for i in range(tiles_cnt_in_v_axis):
        for j in range(tiles_cnt_in_h_axis):
            res_tiles += [get_2d_array_from_bigger_2d_array(entire_combined_rect, left_border + tile_size * j,
                                                            top_border + tile_size * i, tile_size, overlap)]

    return res_tiles, right_shift, bottom_shift, tiles_cnt_in_v_axis, tiles_cnt_in_h_axis


def get_shifted_coords(scale_in_deg, img_array, long_top_left, lat_top_left, long_delta, lat_delta, ncols, count):
    # get width and height of image array
    n_img_rows, n_img_cols = img_array.shape

    # The shift is one tile side length in degrees. It should be float value
    shift = scale_in_deg

    img_row, img_col = get_2d_index_by_element_number(ncols, count)

    long_min_shifted = round(long_top_left + shift * img_col)
    lat_max_shifted = round(lat_top_left - shift * img_row)
    long_max_shifted = round(long_min_shifted + (long_delta * n_img_cols))
    lat_min_shifted = round(lat_max_shifted - (abs(lat_delta) * n_img_rows))

    return long_min_shifted, lat_max_shifted, long_max_shifted, lat_min_shifted


def round_one_dim_array_to_default_scale(init_arr, init_arr_scale_in_deg):
    rounded_arr = []
    init_arr_length = len(init_arr)

    for i in range(0, init_arr_length - 1, int(init_arr_scale_in_deg)):
        average = 0
        for j in range(i, i + int(init_arr_scale_in_deg)):
            average += init_arr[j]
        average /= int(init_arr_scale_in_deg)
        rounded_arr.append(average)

    return rounded_arr


def round_2d_array_to_default_scale(init_arr, round_number):
    rounded_arr = []
    init_arr_side = np.shape(init_arr)[0]

    # rows (vertical direction)
    for row in range(0, init_arr_side - 1, int(round_number)):
        temp_arr = []
        # cols (horizontal direction)
        for col in range(0, init_arr_side - 1, int(round_number)):
            average = 0
            for inner_i in range(row, row + int(round_number)):
                for inner_j in range(col, col + int(round_number)):
                    average += init_arr[inner_i][inner_j]
            average /= int(round_number) ** 2
            temp_arr.append(average)
        temp_arr.append(init_arr[row][init_arr_side - 1])
        rounded_arr += [temp_arr]

    last_rounded_row = round_one_dim_array_to_default_scale(init_arr[init_arr_side - 1], round_number)
    last_rounded_row.append(init_arr[init_arr_side - 1][init_arr_side - 1])
    rounded_arr += [last_rounded_row]

    return np.asarray(rounded_arr)


def replace_nodata_value_in_array(array):
    array[array == 32768] = 0


def plot_image_grid(images, gdal_current_ds, right_shift, bottom_shift, scale_in_deg, res_dict,
                    ncols, nrows, cmap='gist_earth'):
    """Plot a grid of images"""
    # print("^", ncols, nrows)

    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    # f, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # axes = axes.flatten()[:len(imgs)]

    count = 1
    default_scale_in_px = 1200
    long_min, long_delta, dxdy, lat_max, dydx, lat_delta = gdal_current_ds.GetGeoTransform()

    long_left = long_min + (right_shift / default_scale_in_px)
    lat_top = lat_max - (bottom_shift / default_scale_in_px)

    for img in imgs:
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()

            replace_nodata_value_in_array(img)

            long_min_shifted, lat_max_shifted, long_max_shifted, lat_min_shifted = get_shifted_coords(
                scale_in_deg, img, long_left, lat_top,
                long_delta, lat_delta, ncols, count)
            count += 1

            # Here I need to round img if it has scale more than min scale (1 deg)
            if int(scale_in_deg) > 1:
                img = round_2d_array_to_default_scale(img, scale_in_deg)

            res_dict[(lat_max_shifted, lat_min_shifted, long_min_shifted, long_max_shifted)] = img

    # for img, ax in zip(imgs, axes.flatten()):
    #     if np.any(img):
    #         if len(img.shape) > 2 and img.shape[2] == 1:
    #             img = img.squeeze()
    #
    #         ax.title.set_text(count)
    #         long_min_shifted, lat_max_shifted, long_max_shifted, lat_min_shifted = get_shifted_coords(
    #             scale_in_deg, img, long_left, lat_top,
    #             long_delta, lat_delta, ncols, count)
    #
    #         # Here I need to round img if it has scale more than min scale (1 deg)
    #         if int(scale_in_deg) > 1:
    #             img = round_2D_array_to_default_scale(img, scale_in_deg)
    #
    #         ax.imshow(img, cmap=cmap, extent=[long_min_shifted, long_max_shifted, lat_min_shifted, lat_max_shifted])
    #         count += 1
    #
    #         res_dict[(lat_max_shifted, lat_min_shifted, long_min_shifted, long_max_shifted)] = img
    #
    # f.show()


def fill_dict_with_res_array(init_tiles, gdal_current_ds, right_shift, bottom_shift, scale_in_deg, res_dict,
                             ncols, nrows):

    tiles = [init_tiles[i] if len(init_tiles) > i else None for i in range(nrows * ncols)]

    count = 1
    default_scale_in_px = 1200
    long_min, long_delta, dxdy, lat_max, dydx, lat_delta = gdal_current_ds.GetGeoTransform()

    long_left = long_min + (right_shift / default_scale_in_px)
    lat_top = lat_max - (bottom_shift / default_scale_in_px)

    for tile in tiles:
        replace_nodata_value_in_array(tile)

        long_min_shifted, lat_max_shifted, long_max_shifted, lat_min_shifted = get_shifted_coords(
            scale_in_deg, tile, long_left, lat_top,
            long_delta, lat_delta, ncols, count)
        count += 1

        # Here I need to round img if it has scale more than min scale (1 deg)
        if int(scale_in_deg) > 1:
            tile = round_2d_array_to_default_scale(tile, scale_in_deg)

        res_dict[(lat_max_shifted, lat_min_shifted, long_min_shifted, long_max_shifted)] = tile


def get_tile_arrays_and_coords_by_grid_coords(h_start, h_cnt, v_start, v_cnt, total_counter):
    tile_arrays_by_grid_coords = {}
    tile_geodata_by_grid_coords = {}

    for v in range(v_start, v_start + v_cnt):
        for h in range(h_start, h_start + h_cnt):
            v_num = re.sub(r"^(\d+)", pad_number2, str(v))
            h_num = re.sub(r"^(\d+)", pad_number2, str(h))
            grid_coords = h_num + '_' + v_num

            path_to_file = 'C:/Users/tum/Programming/SRTM_data/srtm_data/' + grid_coords + \
                           '/srtm_' + grid_coords + '.tif'

            if os.path.isfile(path_to_file):
                print(path_to_file)
                total_counter.increment()

                ds = gdal.Open(path_to_file)
                band = ds.GetRasterBand(1)
                elevations = band.ReadAsArray()
                tile_arrays_by_grid_coords[grid_coords] = elevations
                # print('*')
                # tile_arrays_by_grid_coords[grid_coords] = np.asarray(
                #     get_2d_array_from_bigger_2d_array(elevations, 0, 0, 6000, 0))
                tile_geodata_by_grid_coords[grid_coords] = ds
                # print('/')

    return tile_arrays_by_grid_coords, tile_geodata_by_grid_coords


# Start of execution

def main(scale_in_sec, h_start, h_cnt, v_start, v_cnt, total_counter):
    gdal.UseExceptions()

    rects = [[], [], [], []]
    cap_array = np.full((6000, 6000), 0)

    srtm_tile_h_cnt = 72
    srtm_tile_v_cnt = 24

    right_shift, bottom_shift, prev_right_shift, prev_bottom_shift = 0, 0, 0, 0

    scale_in_deg, rect_side_in_px = get_scale_in_degrees_and_rect_side_in_px(scale_in_sec)

    tile_arrays_by_grid_coords, tile_geodata_by_grid_coords = \
        get_tile_arrays_and_coords_by_grid_coords(h_start, h_cnt, v_start, v_cnt, total_counter)

    tiles_by_coords = {}

    for v in range(v_start, v_start + v_cnt + 1):
        prev_bottom_shift = bottom_shift
        prev_right_shift = 0
        for h in range(h_start, h_start + h_cnt + 1):
            current_rect_row = re.sub(r"^(\d+)", pad_number2, str(v))
            current_rect_col = re.sub(r"^(\d+)", pad_number2, str(h))
            current_rect_coords = current_rect_col + '_' + current_rect_row

            if current_rect_coords in tile_arrays_by_grid_coords:
                rects[0] = tile_arrays_by_grid_coords[current_rect_coords]
                rects[1] = get_right_top_rect(h, v, srtm_tile_h_cnt, tile_arrays_by_grid_coords, cap_array)
                rects[2] = get_left_bottom_rect(h, v, srtm_tile_v_cnt, tile_arrays_by_grid_coords, cap_array)
                rects[3] = get_right_bottom_rect(h, v, srtm_tile_v_cnt, srtm_tile_h_cnt,
                                                 tile_arrays_by_grid_coords, cap_array)

                tiles, right_shift, bottom_shift, nrows, ncols = \
                    split_rect_in_defined_tiles(rects, int(rect_side_in_px), prev_right_shift, prev_bottom_shift, 1)
                fill_dict_with_res_array(np.array(tiles), tile_geodata_by_grid_coords[current_rect_coords],
                                         prev_right_shift, prev_bottom_shift, scale_in_deg, tiles_by_coords, ncols, nrows)
                prev_right_shift = right_shift

    return tiles_by_coords


def get_filename_by_coords(lat, lon):
    filename = "N" if lat >= 0 else "S"
    filename += re.sub(r"^(\d+)", pad_number2, str(abs(lat)))
    filename += "E" if lon >= 0 else "W"
    filename += re.sub(r"^(\d+)", pad_number3, str(abs(lon)))
    filename += '.hgt'

    return filename


def write_hgt_file(res_dict, total_counter, progress_counter):
    for coords, img in res_dict.items():
        filename = get_filename_by_coords(coords[1], coords[2])
        img_size = np.shape(res_dict[coords])[0]

        with open('D:/ReliefProject/res/6x6/N61-60/' + filename, 'wb') as f:
            row = np.zeros(img_size * 2, dtype=np.int8)
            for lat_step in range(0, img_size):
                # row = np.full(img_size * 2, 0, dtype=np.int8)
                for lon_step in range(0, img_size):
                    m_next = np.int16(img[lat_step, lon_step])
                    row[2 * lon_step] = m_next >> 8
                    row[2 * lon_step + 1] = (m_next & 0xFF)

                f.write(row.astype('int8').tobytes())
        progress_counter.increment()
        pb.printProgressBar(progress_counter.get(), total_counter.get() * 25, prefix='Progress:', suffix='Complete',
                            length=50)


def read_hgt_file(path_to_file):
    res_arr = np.full((1201, 1201), 0)
    img_size = 1201
    with open(path_to_file, 'rb') as f:
        row = np.full(img_size * 2, 0, dtype=np.int8)
        for lat_step in range(0, img_size):
            row = f.read(img_size * 2)
            for lon_step in range(0, img_size):
                m_next = np.int16()
                high = row[2 * lon_step]
                low = np.uint8(row[2 * lon_step + 1])
                m_next = high * 256 + low if high >= 0 else -((-high * 256) + (-low))
                res_arr[lat_step, lon_step] = m_next


    return res_arr
    # plt.imshow(res_arr, cmap='gist_earth')
    # plt.show()


def write_one_hgt_file(path_to_file, data):
    data_size = np.shape(data)[0]

    with open(path_to_file, 'wb') as f:
        row = np.zeros(data_size * 2, dtype=np.int8)
        for lat_step in range(0, data_size):
            for lon_step in range(0, data_size):
                m_next = np.int16(data[lat_step, lon_step])
                row[2 * lon_step] = m_next >> 8
                row[2 * lon_step + 1] = (m_next & 0xFF)

            f.write(row.astype('int8').tobytes())


def replace_value_in_hgt_file(path_to_file):
    res_arr = np.full((1201, 1201), 0)
    img_size = 1201
    with open(path_to_file, 'rb') as f:
        row = np.full(img_size * 2, 0, dtype=np.int8)
        for lat_step in range(0, img_size):
            row = f.read(img_size * 2)
            for lon_step in range(0, img_size - 1):
                m_next = np.int16()
                high = row[2 * lon_step]
                low = np.uint8(row[2 * lon_step + 1])
                m_next = high * 256 + low if high >= 0 else -((-high * 256) + (-low))
                res_arr[lat_step, lon_step] = m_next

        res_arr[res_arr == 32768] = 0
        write_one_hgt_file(path_to_file, res_arr)


def replace_value_in_files_in_dir(dest_dir):
    count = 0
    total = 900
    for filename in os.listdir(dest_dir):
        replace_value_in_hgt_file(os.path.join(dest_dir, filename))
        count += 1
        print('\r' + str(count) + '/' + str(total) + ' files')

