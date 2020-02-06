#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:30:59 2020

@author: c4rl0s
"""

from pymongo import MongoClient
import pandas as pd
import turicreate as tc

client = MongoClient('localhost', 27017)
db = client['shopping-cart']
collection = db['recoms']

exclude_data = {'_id': False, '__v': False}
raw_data = list(collection.find({}, projection=exclude_data))
data_df = pd.DataFrame(raw_data)
print(data_df)

data = tc.SFrame(data_df)

model = tc.factorization_recommender.create(
    data, 'userId', 'itemId', target='rating')


results = model.recommend()
model.save("my_model.model")
print(results)



