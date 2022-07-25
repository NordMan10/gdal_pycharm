import gdal
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import re


def test1():
    print("Hello")


def pad_number(match):
    number = int(match.group(1))
    return format(number, "02d")


# Functions that return rectangles near to a current rectangle

def get_right_top_rect(current_col, current_row, srtm_tile_h_cnt, tile_arrays_by_grid_coords, cap_array):
    right_top_rect_coords = ""
    if current_col == srtm_tile_h_cnt:
        right_top_rect_coords = re.sub(r"^(\d+)", pad_number, str(1)) + "_" + re.sub(r"^(\d+)", pad_number,
                                                                                     str(current_row))
    else:
        right_top_rect_coords = re.sub(r"^(\d+)", pad_number, str(current_col + 1)) + "_" + re.sub(r"^(\d+)",
                                                                                                   pad_number,
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
        left_bottom_rect_coords = re.sub(r"^(\d+)", pad_number, str(current_col)) + "_" + re.sub(r"^(\d+)",
                                                                                                 pad_number,
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
        right_bottom_rect_coords = re.sub(r"^(\d+)", pad_number, str(1)) + "_" + re.sub(r"^(\d+)",
                                                                                        pad_number,
                                                                                        str(current_row + 1))
    else:
        right_bottom_rect_coords = re.sub(r"^(\d+)", pad_number, str(current_col + 1)) + "_" + re.sub(r"^(\d+)",
                                                                                                      pad_number,
                                                                                                      str(current_row + 1))

    if right_bottom_rect_coords in tile_arrays_by_grid_coords:
        return tile_arrays_by_grid_coords[right_bottom_rect_coords]
    else:
        return cap_array


def get_scale_in_degrees_and_rect_side_in_px(seconds_cnt):
    # constants
    min_scale = 3
    default_pixel_num = 1200
    overlap_pixel_num = 1

    scale_in_degrees = seconds_cnt / min_scale
    rect_side = default_pixel_num * scale_in_degrees

    return scale_in_degrees, rect_side


# The function takes big 2D array and returns smaller 2D array in according to passed sizes
# If the function can't satisfy passed sizes of array, it returns array with less sizes
def get_2D_array_from_bigger_2D_array(init_arr, left_border, top_border,
                                      tile_size, overlap):
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
    element_col = 0

    while number > ncols:
        number -= ncols
        element_row += 1

    element_col = number - 1

    return (element_row, element_col)


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
            res_tiles += [get_2D_array_from_bigger_2D_array(entire_combined_rect, left_border + tile_size * j,
                                                            top_border + tile_size * i, tile_size, overlap)]

    return (res_tiles, right_shift, bottom_shift, tiles_cnt_in_v_axis, tiles_cnt_in_h_axis)


def get_georeferencing_data(scale_in_deg, img_array, long_top_left, lat_top_left, long_delta, lat_delta, ncols, count):
    # get width and height of image array
    n_img_rows, n_img_cols = img_array.shape

    # The shift is one tile side length in degrees. It's should be float value
    shift = scale_in_deg

    img_row, img_col = get_2d_index_by_element_number(ncols, count)

    long_min_shifted = round(long_top_left + shift * img_col)
    lat_max_shifted = round(lat_top_left - shift * img_row)
    long_max_shifted = round(long_min_shifted + (long_delta * n_img_cols))
    lat_min_shifted = round(lat_max_shifted - (abs(lat_delta) * n_img_rows))

    return (long_min_shifted, lat_max_shifted, long_max_shifted, lat_min_shifted)


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


def round_2D_array_to_default_scale(init_arr, init_arr_scale_in_deg):
    rounded_arr = []
    init_arr_side = np.shape(init_arr)[0]

    # rows (vertical direction)
    for i in range(0, init_arr_side - 1, int(init_arr_scale_in_deg)):
        temp_arr = []
        # cols (horizontal direction)
        for j in range(0, init_arr_side - 1, int(init_arr_scale_in_deg)):
            average = 0
            for inner_i in range(i, i + int(init_arr_scale_in_deg)):
                for inner_j in range(j, j + int(init_arr_scale_in_deg)):
                    average += init_arr[inner_i][inner_j]
            average /= int(init_arr_scale_in_deg) ** 2
            temp_arr.append(average)
        temp_arr.append(init_arr[i][init_arr_side - 1])
        rounded_arr += [temp_arr]

    last_rounded_row = round_one_dim_array_to_default_scale(init_arr[init_arr_side - 1], init_arr_scale_in_deg)
    last_rounded_row.append(init_arr[init_arr_side - 1][init_arr_side - 1])
    rounded_arr += [last_rounded_row]

    return np.asarray(rounded_arr)


def plot_image_grid(images, gdal_current_ds, right_shift, bottom_shift, scale_in_deg, res_dict,
                    ncols, nrows, cmap='gist_earth'):
    """Plot a grid of images"""
    #     if not ncols:
    #     factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
    #     ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    #     nrows = int(len(images) / ncols) + int(len(images) % ncols)
    print("^", ncols, nrows)
    #     ncols = int(math.sqrt(np.shape(images)[0]))
    #     nrows = int(math.sqrt(np.shape(images)[1]))
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()[:len(imgs)]

    count = 1
    default_scale_in_deg = 5
    default_scale_in_px = 1200
    long_min, long_delta, dxdy, lat_max, dydx, lat_delta = gdal_current_ds.GetGeoTransform()

    long_top_left = long_min + (right_shift / default_scale_in_px)
    lat_top_left = lat_max - (bottom_shift / default_scale_in_px)

    for img, ax in zip(imgs, axes.flatten()):
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()

            ax.title.set_text(count)
            long_min_shifted, lat_max_shifted, long_max_shifted, lat_min_shifted = get_georeferencing_data(
                scale_in_deg, img, long_top_left, lat_top_left,
                long_delta, lat_delta, ncols, count)

            # Here I need to round img if it has scale more than min scale (1 deg)
            #             if(int(scale_in_deg) > 1):
            #                 img = round_2D_array_to_default_scale(img, scale_in_deg)

            #             print("*", img.shape)
            ax.imshow(img, cmap=cmap, extent=[long_min_shifted, long_max_shifted, lat_min_shifted, lat_max_shifted])
            count += 1

            res_dict[(lat_max_shifted, lat_min_shifted, long_min_shifted, long_max_shifted)] = img

    f.show()


# Start of execution


def main():
    gdal.UseExceptions()

    # gdal_ds = []
    srtm_tile_h_cnt = 72
    h_start = 50
    srtm_tile_v_cnt = 5

    tile_arrays_by_grid_coords = {}
    tile_geodata_by_grid_coords = {}
    res_dict = {}

    for v in range(srtm_tile_v_cnt, srtm_tile_v_cnt + 2):
        for h in range(h_start, srtm_tile_h_cnt + 1):
            v_num = re.sub(r"^(\d+)", pad_number, str(v))
            h_num = re.sub(r"^(\d+)", pad_number, str(h))

            grid_coords = h_num + '_' + v_num
            path_to_dir = 'C:/Users/tum/Programming/srtm_data/' + grid_coords

            print(path_to_dir)

            path_to_file = path_to_dir + '/srtm_' + grid_coords + '.tif'

            if os.path.isdir(path_to_dir):
                if os.path.isfile(path_to_file):
                    ds = gdal.Open(path_to_file)
                    band = ds.GetRasterBand(1)
                    elevations = band.ReadAsArray()
                    tile_arrays_by_grid_coords[grid_coords] = elevations
                    tile_geodata_by_grid_coords[grid_coords] = ds

    rects = [[], [], [], []]
    cap_array = np.full((6000, 6000), -32768)

    right_shift = 0
    bottom_shift = 0

    prev_right_shift = 0
    prev_bottom_shift = 0

    seconds_cnt = 6
    scale_in_deg, rect_side_in_px = get_scale_in_degrees_and_rect_side_in_px(seconds_cnt)

    for v in range(1, srtm_tile_v_cnt + 1):
        prev_bottom_shift = bottom_shift
        for h in range(1, srtm_tile_h_cnt + 1):
            current_rect_row = re.sub(r"^(\d+)", pad_number, str(v))
            current_rect_col = re.sub(r"^(\d+)", pad_number, str(h))
            current_rect_coords = current_rect_col + '_' + current_rect_row

            # 2.2
            if current_rect_coords in tile_arrays_by_grid_coords:
                rects[0] = tile_arrays_by_grid_coords[current_rect_coords]
                # 2.3
                rects[1] = get_right_top_rect(h, v, srtm_tile_h_cnt, tile_arrays_by_grid_coords, cap_array)
                rects[2] = get_left_bottom_rect(h, v, srtm_tile_v_cnt, tile_arrays_by_grid_coords, cap_array)
                rects[3] = get_right_bottom_rect(h, v, srtm_tile_v_cnt, srtm_tile_h_cnt,
                                                 tile_arrays_by_grid_coords, cap_array)

                # 2.4
                tiles, right_shift, bottom_shift, nrows, ncols = split_rect_in_defined_tiles(rects,
                                                                                             int(rect_side_in_px),
                                                                                             prev_right_shift,
                                                                                             prev_bottom_shift, 1)
                plot_image_grid(np.array(tiles), tile_geodata_by_grid_coords[current_rect_coords],
                                prev_right_shift, prev_bottom_shift,
                                scale_in_deg, res_dict, ncols, nrows)
                prev_right_shift = right_shift
