import os.path as osp
import mmcv
import numpy as np
import json
import os
import cv2
import pycocotools.mask as maskUtils

def convert_maskrcnn_to_coco(ann_file, out_file, image_prefix):
    '''
    ann_file: 原始标注文件
    out_file: 输出的json文件，转换为coco数据集形式
    image_prefix: 图片所在文件夹
    '''
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    category = []
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        image_id = v['image_id']
        filename = '0000'+str(image_id)+'.jpg'
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bbox = v['bbox']
        score = v['score']
        category_id = v['category_id']

        seg = v['segmentation']
        [h, w] = seg['size']
        rle = seg['counts']
        # segmentation = coco.annToMask(rle)
        mask = maskUtils.decode(rle)
        _, contours, _ = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                segmentation.append(contour)

        annotations = dict(
            id=idx,
            image_id=image_id,
            category_id=category_id,
            segmentation=segmentation,
            area=h*w,
            bbox=bbox,
            iscrowd=0,
            score=score)

        category = dict(
            id=category_id,
            name=CLASSES[category_id]
        )

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=category)
    mmcv.dump(coco_format_json, out_file)

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

if __name__ == '__main__':
    ann_file = 'maskrcnn_pred.json'
    out_file = 'annotation_maskrcnn.json'
    image_prefix = 'data/coco/train2017/'
    convert_maskrcnn_to_coco(ann_file, out_file, image_prefix)