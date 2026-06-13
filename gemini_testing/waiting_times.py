print("Importing libraries...")
from datetime import datetime
import numpy as np
import time
print("Imports done")

SIMULATE = True
n_reqs = 10
MAX_REQS_PER_MIN = 15
avg_waiting_time = np.ceil(60/MAX_REQS_PER_MIN)

# seed will be yyyymmddhhmmss at execution
now = datetime.now()
seed_str = (
        str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) +
        str(now.hour).zfill(2) + str(now.minute).zfill(2) +
        str(now.second).zfill(2)
    )
print("Random seed:", seed_str)
seed_int = int(seed_str)

rng = np.random.default_rng(seed=seed_int)
waiting_times = np.zeros(n_reqs)
while (
    (np.convolve(waiting_times, np.ones(MAX_REQS_PER_MIN), mode="valid")
     <= 60).any()
    # if any MAX_REQS_PER_MIN consecutive requests would occur under a minute
    ):
    # then up the average time...
    avg_waiting_time += 1
    # and reroll the dice
    waiting_times = rng.exponential(
            scale=avg_waiting_time,
            size=n_reqs
        )

print("Planned average waiting time:", avg_waiting_time)
print("Planned waiting times:")
print(waiting_times)
print("Total waiting time:", waiting_times.sum())

if SIMULATE:
    print("Simulating...")
    for idx, wt in enumerate(waiting_times):
        # print(f"Waiting for {idx+1}...")
        time.sleep(wt)
        print(f"{idx+1} arrived!")

print("All done")
