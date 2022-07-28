import gdal
import numpy as np
import math
import matplotlib.pyplot as plt
# import os
# import re
import threading

import get_relief_data as relief
# import downdoad_srtm_data as d_srtm

relief.main()


# thread1 = threading.Thread(target=d_srtm.download_srtm_data, args=[1, 72, 22])
# thread2 = threading.Thread(target=d_srtm.download_srtm_data, args=[1, 72, 23])
# thread3 = threading.Thread(target=d_srtm.download_srtm_data, args=[1, 72, 24])
# thread1.start()
# thread2.start()
# thread3.start()
