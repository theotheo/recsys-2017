'''
Baseline solution for the ACM Recsys Challenge 2017
using XGBoost

by Daniel Kohlsdorf
'''



import numpy as np
import xgboost as xgb

from model import Interaction, Item, User
from parser import InteractionBuilder, select

# import random

print(" --- Recsys Challenge 2017 Baseline --- ")


USERS_FILE = "users.csv"
ITEMS_FILE = "items.csv"
INTERACTIONS_FILE = "interactions.csv"


'''
1) Parse the challenge data, exclude all impressions
   Exclude all impressions
'''
(header_users, users) = select(USERS_FILE,
                               lambda x: True, build_user, lambda x: int(x[0]))
(header_items, items) = select(ITEMS_FILE,
                               lambda x: True, build_item, lambda x: int(x[0]))

builder = InteractionBuilder(users, items)
(header_interactions, interactions) = select(
    INTERACTIONS_FILE,
    lambda x: x[2] != '0',
    builder.build_interaction,
    lambda x: (int(x[0]), int(x[1]))
)


'''
2) Build recsys training data
'''
data = np.array([interactions[key].features() for key in interactions.keys()])
labels = np.array([interactions[key].label() for key in interactions.keys()])
dataset = xgb.DMatrix(data, label=labels)
dataset.save_binary("recsys2017.buffer")


'''
3) Train XGBoost regression model with maximum tree depth of 2 and 25 trees
'''
evallist = [(dataset, 'train')]
param = {'bst:max_depth': 2,
         'bst:eta': 0.1,
         'silent': 1,
         'objective': 'reg:linear',
         }

param.update({
    'nthread': 4,
    'eval_metric': 'rmse',
    'base_score': 0.0
})
print(param)
num_round = 25
bst = xgb.train(param, dataset, num_round, evallist)
bst.save_model('recsys2017.model')
