
import multiprocessing

import numpy as np
import xgboost as xgb

from parser import build_item, build_user, select
from recommendation_worker import classify_worker

N_WORKERS = 2

USERS_FILE = "users.csv"
ITEMS_FILE = "items.csv"
INTERACTIONS_FILE = "interactions.csv"

TARGET_USERS = "targetUsers.csv"
TARGET_ITEMS = "targetItems.csv"


'''
1) Parse the challenge data, exclude all impressions
   Exclude all impressions
'''
(header_users, users) = select(USERS_FILE,
                               lambda x: True, build_user, lambda x: int(x[0]))
(header_items, items) = select(ITEMS_FILE,
                               lambda x: True, build_item, lambda x: int(x[0]))


'''
4) Create target sets for items and users
'''
target_users = []
for line in open(TARGET_USERS):
    target_users += [int(line.strip())]
target_users = set(target_users)

target_items = []
for line in open(TARGET_ITEMS):
    target_items += [int(line.strip())]


'''
5) Schedule classification
'''
bst = xgb.Booster({'nthread': 4})
bst.load_model('recsys2017.model')
bucket_size = len(target_items) / N_WORKERS
print(bucket_size)
# import ipdb; ipdb.set_trace()
start = 0
jobs = []
# print(items[:5])
# print(users[:5])
# print(target_items)
# print(target_users)
for i in range(0, N_WORKERS):
    stop = int(min(len(target_items), start + bucket_size))
    filename = "solution_" + str(i) + ".csv"
    process = multiprocessing.Process(target=classify_worker, args=(
        target_items[start:stop], target_users, items, users, filename, bst))
    jobs.append(process)
    start = stop

for j in jobs:
    j.start()

for j in jobs:
    j.join()
