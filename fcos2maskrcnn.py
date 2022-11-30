import json
import os
import mmcv

ann_file = r"r50_fcos_top50_det_results.json"
data_infos = mmcv.load(ann_file)
for key, value in data_infos.items():
    if key == '108853':
        print(value) # 所有的标注 list
        print(value[0]) # 第一个标注信息 有image_id,bbox,score,category_id 是dict
        bbox = [obj['bbox'] for obj in value]
        print(bbox)


        exit()
    # image_id = value['image_id']
    # assert key == image_id
