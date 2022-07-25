# importing necessary modules
import requests
import zipfile as zf
from io import BytesIO
import re
import threading


def get_number_in_zeros_format(match):
    number = int(match.group(1))
    return format(number, "02d")


def download_one_piece_srtm_data(v, h):

    v_num = re.sub(r"^(\d+)", get_number_in_zeros_format, str(v))
    h_num = re.sub(r"^(\d+)", get_number_in_zeros_format, str(h))

    dir_name = h_num + '_' + v_num

    print('Downloading srtm_' + dir_name + '.tif started')
    url = 'https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_' + dir_name + '.zip'

    # Downloading the file by sending the request to the URL
    req = requests.get(url)

    # Split URL to get the dest_path
    zip_filename = url.split('/')[-1]
    # temp_dir_name = (zip_filename.split('_')[1 : 3])
    # dir_name = '_'.join(temp_dir_name).split('.')[0]
    dest_path = 'C:/Users/tum/Programming/srtm_data/' + dir_name + '/'

    print('Downloading srtm_' + dir_name + '.tif completed')

    # extracting the zip file contents
    zipfile = zf.ZipFile(BytesIO(req.content))
    zipfile.extractall(dest_path)

    print('Extracting srtm_' + dir_name + '.tif completed')
    print('==========================')


def download_srtm_data(start, srtm_tile_h_cnt, srtm_tile_v_cnt):
    # srtm_tile_h_cnt = 24
    # srtm_tile_v_cnt = 1

    # thread = threading.Thread(target=download_one_piece_srtm_data, args=[v, h])

    for v in range(srtm_tile_v_cnt, srtm_tile_v_cnt + 1):
        # thread = threading.Thread(target=download_one_piece_srtm_data, args=[v, h])
        for h in range(start, start + srtm_tile_h_cnt):

            # thread.start()
            try:
                download_one_piece_srtm_data(v, h)
            except BaseException:
                print("The tile is not exist (most likely)")
                print('==========================')
                pass
