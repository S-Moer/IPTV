#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# 抽取5.1-5.30电视剧数据cut
# data_path = 'data'
# files = os.listdir(data_path)
#
# data = pd.read_csv(data_path+'behavior_20210501.csv', error_bad_lines=False)
# cut = data.loc[data['scene_id'] == 'main_page_series_v7']
#
# for f in tqdm(files):
#     if (f == 'behavior_20210501.csv' or f == 'program_20210727.csv' or f == 'behavior_20210531.csv'):
#         continue
#     data = pd.read_csv(data_path + '/' + f, error_bad_lines=False)
#     temp = data.loc[data['scene_id'] == 'main_page_series_v7']
#     cut = pd.concat([cut, temp])
# cut =  cut.reset_index(drop=True)

#处理豆瓣数据集
data_path = 'moviedata-10m/moviedata-10m/movies.csv'
douban_pg = pd.read_csv(data_path, error_bad_lines=False)
name_list = douban_pg["NAME"]
genres_list = douban_pg["GENRES"]
genre_dict = {}
length = len(name_list)
for i in tqdm(range(length)):
    genre_dict[name_list[i]] = genres_list[i]
#genre_dict格式：{'情定河州':'剧情/爱情'}


#字典形式存储节目信息
data_path = 'data/'
pg = pd.read_csv(data_path+'program_20210727.csv', error_bad_lines=False)
a = 1
content_name_list = pg["content_name"]
genre_list = pg["genre"]

length = len(genre_list) #1679075
genre_new_list = []
for i in range(length):
    genre_new_list.append(i)
#
#
#
# pg[:,1] = genre_new_list
# aaa = 1
genre_new_list = []
cnt = 0
for idx in tqdm(range(len(content_name_list))):
    content_name = content_name_list[idx]
    content_str_list = content_name.split("-")
    flag = 0
    for content_str in content_str_list:
        if content_str in genre_dict:
            cnt = cnt + 1
            #print(content_str)
            #print(genre_dict[content_str])
            genre_new_list.append(genre_dict[content_str])
            flag = 1
            break
    if flag == 0:
        genre_new_list.append(genre_list[idx])

f = open("ans.txt", "w")
print(len(genre_new_list))
print(cnt)
for i in genre_new_list:
    f.writelines(str(i))
    f.writelines("\n")
#需要把genre_new_list更新到pg中，然后保存下就行了
# pg[:,0] = genre_new_list
# aaa = 1


# temp = pg[pg['item_code'].str.contains('seriespackage')]
# temp = temp.drop_duplicates(subset=['item_code'],keep='first',inplace=False)
# temp = temp.drop(temp[temp['is_rec'] == 0].index)
# temp = temp.reset_index(drop=True)
#
# item = temp[['item_code','is_rec']]
# item.set_index(keys='item_code', inplace=True)
# item = item.T
# dic = item.to_dict(orient='records')[0]
#
# ans = pd.DataFrame(columns=('log_datetime','session_id','user_id','scene_id','strategy_id','req_entity_id','return_pos_id','return_item_id','is_exposure','is_click','is_play'))
#
# #过滤节目单中没有的节目，分成75份数据，每份1万条
# for i in range(0,75):
#     print(i)
#     ans1 = pd.DataFrame(columns=(
#     'log_datetime', 'session_id', 'user_id', 'scene_id', 'strategy_id', 'req_entity_id', 'return_pos_id',
#     'return_item_id', 'is_exposure', 'is_click', 'is_play'))
#     for j in tqdm(range(i*10000,i*10000+10000)):
#         item = cut.loc[j]['return_item_id']
#         if item in dic:
#             t = cut.loc[j].T
#             ans1 = ans1.append([t])
#     ans = pd.concat([ans, ans1])
#
# ans1 = pd.DataFrame(columns=(
#     'log_datetime', 'session_id', 'user_id', 'scene_id', 'strategy_id', 'req_entity_id', 'return_pos_id',
#     'return_item_id', 'is_exposure', 'is_click', 'is_play'))
#
# #最后几千条数据
# for j in tqdm(range(750000,759659)):
#     item = cut.loc[j]['return_item_id']
#     if item in dic:
#         t = cut.loc[j].T
#         ans1 = ans1.append([t])
#
# ans = pd.concat([ans, ans1])
# ans =  ans.reset_index(drop=True)
# ans.to_csv('ans.csv', sep= ' ',index=0)