from osgeo import gdal
import os
import get_relief_data as relief
import re
import numpy as np
import progress_bar as pb
import math


def pad_number2(match):
    number = int(match.group(1))
    return format(number, "02d")


def pad_number3(match):
    number = int(match.group(1))
    return format(number, "03d")


def get_right_top_rect(pole, world_side, current_lat, current_lon, cap_array, max_lat, min_lat):
    right_top_rect_coords = ""
    if world_side == "E" and current_lon == 180:
        return cap_array
    if world_side == "W" and current_lon == 1:
        right_top_rect_coords = pole + re.sub(r"^(\d+)", pad_number2, str(current_lat)) + "E" + \
                                re.sub(r"^(\d+)", pad_number3, str(0))
    elif world_side == "W":
        right_top_rect_coords = pole + re.sub(r"^(\d+)", pad_number2, str(current_lat)) + world_side + \
                                re.sub(r"^(\d+)", pad_number3, str(current_lon - 1))
    elif world_side == "E":
        right_top_rect_coords = pole + re.sub(r"^(\d+)", pad_number2, str(current_lat)) + world_side + \
                                re.sub(r"^(\d+)", pad_number3, str(current_lon + 1))

    path_to_file = get_path_to_file_by_coords(right_top_rect_coords, max_lat, min_lat)
    if os.path.isfile(path_to_file):
        return get_tile_array_by_filename(path_to_file)
    else:
        return cap_array


def get_left_bottom_rect(pole, world_side, current_lat, current_lon, cap_array, max_lat, min_lat):
    left_bottom_rect_coords = ""

    if pole == "S" and current_lat == 83:
        return cap_array
    if pole == "N" and current_lat == 0:
        left_bottom_rect_coords = 'S' + re.sub(r"^(\d+)", pad_number2, str(1)) + world_side + \
                                re.sub(r"^(\d+)", pad_number3, str(current_lon))
    elif pole == "N":
        left_bottom_rect_coords = pole + re.sub(r"^(\d+)", pad_number2, str(current_lat - 1)) + world_side + \
                                re.sub(r"^(\d+)", pad_number3, str(current_lon))
    elif pole == "S":
        left_bottom_rect_coords = pole + re.sub(r"^(\d+)", pad_number2, str(current_lat + 1)) + world_side + \
                                re.sub(r"^(\d+)", pad_number3, str(current_lon))

    path_to_file = get_path_to_file_by_coords(left_bottom_rect_coords, max_lat, min_lat)
    if os.path.isfile(path_to_file):
        return get_tile_array_by_filename(path_to_file)
    else:
        return cap_array


def get_right_bottom_rect(pole, world_side, current_lat, current_lon, cap_array, max_lat, min_lat):

    if (pole == "S" and current_lat == 83) or (world_side == "E" and current_lon == 180):
        return cap_array

    right_bottom_rect_coords = ""
    lat = ''
    lon = ''
    if pole == "N":
        if current_lat == 0:
            lat = 'S' + re.sub(r"^(\d+)", pad_number2, str(1))
        else:
            lat = pole + re.sub(r"^(\d+)", pad_number2, str(current_lat - 1))
    if world_side == "W":
        if current_lon == 1:
            lon = 'E' + re.sub(r"^(\d+)", pad_number3, str(0))
        else:
            lon = world_side + re.sub(r"^(\d+)", pad_number3, str(current_lon + 1))

    right_bottom_rect_coords = lat + lon

    path_to_file = get_path_to_file_by_coords(right_bottom_rect_coords, max_lat, min_lat)
    if os.path.isfile(path_to_file):
        return get_tile_array_by_filename(path_to_file)
    else:
        return cap_array


# In rects there are 2d arrays with overlap pixels
def combine_rects(rects):
    entire_combined_rect = []
    row_combined_rects = []
    loop_step = int(math.sqrt(len(rects)))
    last_row_index = len(rects) - loop_step
    for row in range(0, len(rects), loop_step):
        if row == last_row_index:
            row_combined_rects.append(rects[row][0: len(rects[row]), 0: -1])
        else:
            row_combined_rects.append(rects[row][0: -1, 0: -1])
        for col in range(1, loop_step):
            array_to_combine = rects[row + col]
            if row != last_row_index:
                array_to_combine = array_to_combine[0: -1]
            if col != loop_step - 1:
                array_to_combine = array_to_combine[0: len(array_to_combine), 0: -1]
            row_combined_rects[len(row_combined_rects) - 1] = \
                np.hstack((row_combined_rects[len(row_combined_rects) - 1], array_to_combine))

    for row in range(len(row_combined_rects)):
        if len(entire_combined_rect) == 0:
            entire_combined_rect.append(row_combined_rects[row])
            continue
        entire_combined_rect[0] = np.vstack((entire_combined_rect[0], row_combined_rects[row]))

    return np.asarray(entire_combined_rect[0])


def round_one_dim_array_to_default_scale(init_arr, round_number):
    rounded_arr = []

    for col in range(0, len(init_arr) - 1, int(round_number)):
        average = 0
        for inner_col in range(col, col + int(round_number)):
            average += init_arr[inner_col]
        average /= int(round_number)
        rounded_arr.append(average)

    # rounded_arr.append(init_arr[len(init_arr) - 1])
    return rounded_arr


def round_2d_array_to_default_scale(init_arr, round_number):
    rounded_arr = []
    init_arr_side = len(init_arr)

    # rows (vertical direction)
    for row in range(0, init_arr_side - 1, int(round_number)):
        temp_arr = []
        # cols (horizontal direction)
        for col in range(0, init_arr_side - 1, int(round_number)):
            average = np.int32(0)
            for inner_row in range(row, row + int(round_number)):
                for inner_col in range(col, col + int(round_number)):
                    average += init_arr[inner_row][inner_col]
            average /= int(round_number ** 2)
            temp_arr.append(average)
        # Add overlap column
        temp_arr.append(init_arr[row][init_arr_side - 1])
        rounded_arr += [temp_arr]

    # Round overlap row
    last_rounded_row = round_one_dim_array_to_default_scale(init_arr[init_arr_side - 1], round_number)
    # Add overlap column for overlap row
    last_rounded_row.append(init_arr[init_arr_side - 1][init_arr_side - 1])
    # Add overlap row
    rounded_arr += [last_rounded_row]

    # return rounded array
    return np.asarray(rounded_arr)


# coords -- geographical coordinates, e.g N69W003
def get_path_to_file_by_coords(coords, max_lat, min_lat):
    return 'D:/ReliefProject/AsterGdemV3Data/data/' + coords[0] + str(max_lat) + '-' + \
                               str(min_lat) + '/ASTGTMV003_' + coords + '_dem.tif'


def get_tile_array_by_filename(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray()


def get_left_bottom_corner_coords_by_path(path_to_file):
    return path_to_file[-15: -8]


def read_one_tif_file_3x3(path_to_file):
    gdal.UseExceptions()

    tile_coords = get_left_bottom_corner_coords_by_path(path_to_file)
    ds = gdal.Open(path_to_file)
    band = ds.GetRasterBand(1)
    elevations = band.ReadAsArray()
    elevations = round_2d_array_to_default_scale(elevations, 3)
    # elevations[elevations == 32768] = 0

    # long_min, long_delta, dxdy, lat_max, dydx, lat_delta = ds.GetGeoTransform()
    # lat_min = lat_max - 1
    # long_max = long_min + 1

    # tiles_by_coords[(lat_max, lat_min, long_min, long_max)] = elevations

    return tile_coords, elevations


def get_one_parallel_tiles(parallel, dest_dir):
    tiles_by_coords = {}
    filenames = os.listdir(dest_dir)
    for filename in filenames:
        if int(filename[12: 14]) == parallel:
            tile_coords, data = read_one_tif_file_3x3(os.path.join(dest_dir + '\\', filename))
            tiles_by_coords[tile_coords] = data

    return tiles_by_coords


def get_tiles_by_coords_3x3(source_dir, dest_dir, progress_counter, total_counter):
    tiles_by_coords = {}

    filenames = os.listdir(source_dir)
    total_count = len(filenames)
    total_counter.add(total_count)
    for filename in filenames:
        path_to_file = os.path.join(source_dir, filename)
        # print(path_to_file)
        tile_coords, data = read_one_tif_file_3x3(path_to_file)
        write_one_hgt_file(tile_coords, data, dest_dir)

        progress_counter.increment()
        pb.printProgressBar(progress_counter.get(), total_counter.get(), prefix='Progress:', suffix='Complete',
                            length=50)

    return tiles_by_coords


def get_tiles_by_coords_6x6(pole, max_lat, min_lat, start_lon, end_lon, total_counter):
    gdal.UseExceptions()
    tiles_by_coords = {}

    cap_array = np.full((3601, 3601), 0)

    sides_of_world = ['W', 'E']
    for lat in range(min_lat, max_lat + 1, 2):
        for lon in range(start_lon, end_lon + 1, 2):
            for side_of_world in sides_of_world:
                path_to_file = 'D:/ReliefProject/AsterGdemV3Data/data/' + pole + str(max_lat) + '-' + \
                               str(min_lat) + '/ASTGTMV003_' + pole + str(lat) + side_of_world + \
                               re.sub(r"^(\d+)", pad_number3, str(lon)) + '_dem.tif'
                if os.path.isfile(path_to_file):
                    print(path_to_file)
                    total_counter.increment()

                    ds = gdal.Open(path_to_file)
                    band = ds.GetRasterBand(1)
                    elevations = band.ReadAsArray()
                    rects = []
                    rects.append(elevations)
                    rects.append(get_right_top_rect(pole, side_of_world, lat, lon, cap_array, max_lat, min_lat))
                    rects.append(get_left_bottom_rect(pole, side_of_world, lat, lon, cap_array, max_lat, min_lat))
                    rects.append(get_right_bottom_rect(pole, side_of_world, lat, lon, cap_array, max_lat, min_lat))

                    elevations = combine_rects(rects)
                    elevations = round_2d_array_to_default_scale(elevations, 6)

                    long_min, long_delta, dxdy, lat_max, dydx, lat_delta = ds.GetGeoTransform()
                    lat_min = lat_max - 2
                    long_max = long_min + 2

                    tiles_by_coords[(lat_max, lat_min, long_min, long_max)] = elevations

    return tiles_by_coords


def create_one_hgt_file_from_aster_tif(path_to_source_file, path_to_dest_dir):
    tile_coords, elevations = read_one_tif_file_3x3(path_to_source_file)
    write_one_hgt_file(tile_coords, elevations, path_to_dest_dir)


def get_filename_by_coords(lat, lon):
    filename = "N" if lat >= 0 else "S"
    filename += re.sub(r"^(\d+)", pad_number2, str(abs(int(lat))))
    filename += "E" if lon >= 0 else "W"
    filename += re.sub(r"^(\d+)", pad_number3, str(abs(int(lon))))
    filename += '.hgt'

    return filename


def write_one_hgt_file(coords, data, dest_dir):
    filename = coords + '.hgt'
    data_size = np.shape(data)[0]

    with open(os.path.join(dest_dir, filename), 'wb') as f:
        row = np.zeros(data_size * 2, dtype=np.int8)
        for lat_step in range(0, data_size):
            for lon_step in range(0, data_size):
                m_next = np.int16(data[lat_step, lon_step])
                row[2 * lon_step] = m_next >> 8
                row[2 * lon_step + 1] = (m_next & 0xFF)

            f.write(row.astype('int8').tobytes())


def write_hgt_files(tiles_dict, dest_dir, total_counter=None, progress_counter=None):
    for coords, data in tiles_dict.items():
        write_one_hgt_file(coords, data, dest_dir)
        if progress_counter is not None:
            progress_counter.increment()
            pb.printProgressBar(progress_counter.get(), total_counter.get(), prefix='Progress:', suffix='Complete',
                                length=50)

