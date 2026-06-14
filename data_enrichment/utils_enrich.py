from datetime import datetime
import numpy as np
from base64 import b64decode
from io import BytesIO
from PIL import Image

def get_datetime_seed():
    now = datetime.now()
    seed_str = (
            str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) +
            str(now.hour).zfill(2) + str(now.minute).zfill(2) +
            str(now.second).zfill(2)
        )
    seed_int = int(seed_str)
    return seed_int

def is_ceiling_window_sum(arr, window_size, the_ceiling):
    # Check if any sum of consecutive values exceed a specified threshold.
    # Exceed => failed ceiling => False.

    sums = np.convolve(arr, np.ones(window_size), mode="valid")

    answer = True
    if (sums > the_ceiling).any():
        answer = False 

    return answer

def is_floor_window_sum(arr, window_size, the_floor):
    # Check if any sum of consecutive values fall below a specified threshold.
    # Below => failed floor => False.

    sums = np.convolve(arr, np.ones(window_size), mode="valid")

    answer = True
    if (sums < the_floor).any():
        answer = False
    
    return answer

def get_floored_waiting_times(
        n_data,
        init_avg=1,
        flooring_window_size=1,
        the_floor=0,
        seed_int=42
    ):
    avg_waiting_time = init_avg
    rng = np.random.default_rng(seed=seed_int)

    waiting_times = rng.exponential(
            scale=avg_waiting_time,
            size=n_data
        )
    success = is_floor_window_sum(
            arr=waiting_times,
            window_size=flooring_window_size,
            the_floor=the_floor
        )

    while (not success):
        # if any MAX_REQS_PER_MIN consecutive requests would occur under a minute
        # then up the average time...
        avg_waiting_time += 1
        # and reroll the dice
        waiting_times = rng.exponential(
                scale=avg_waiting_time,
                size=n_data
            )
        # now check again
        success = is_floor_window_sum(
                arr=waiting_times,
                window_size=flooring_window_size,
                the_floor=the_floor
            )

    return (rng, avg_waiting_time, waiting_times)

def get_PIL_from_base64_text(base64_text):
    trimmed_base64 = base64_text
    if ((len(base64_text) > 23)
        and base64_text[:23] == "data:image/jpeg;base64,"):
        trimmed_base64 = base64_text[23:]

    io_bytes_obj = BytesIO(b64decode(trimmed_base64))
    the_img = Image.open(fp=io_bytes_obj)
    return the_img

