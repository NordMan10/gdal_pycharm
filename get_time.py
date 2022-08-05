from datetime import datetime


def get_current_time():
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    return current_time


def get_time_interval(start_time, end_time):
    FMT = '%H:%M:%S'
    tdelta = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)
    print('\n' + tdelta)

