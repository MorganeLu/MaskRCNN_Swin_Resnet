from pycocotools.coco import COCO
import cv2
import pandas as pd
import json
 
 
def showNimages(image_id, annFile, imageFile, resultFile):
    """
    :param imageidFile: 要查看的图片imageid，存储一列在csv文件里 （目前设计的imageid需要为6位数，如果少于6位数，可以在前面加多个0）
    :param annFile:使用的标注文件
    :param imageFile:要读取的image所在文件夹
    :param resultFile:画了标注之后的image存储文件夹
    :return:
    """
    # data = pd.read_csv(imageidFile)
    # list = data.values.tolist()
    # list = [581781]
    image_id = [image_id]  # 存储的是要提取图片id
    # for i in range(len(list)):
    #     image_id.append(list[i][0])
    # print(image_id)
    # print(len(image_id))
    coco = COCO(annFile)
 
    for i in range(len(image_id)):
        image = cv2.imread(imageFile + '000000' + str(image_id[i]) + '.jpg')
        annIds = coco.getAnnIds(imgIds=image_id[i], iscrowd=None)
        anns = coco.loadAnns(annIds)
        print("bbox number gt: " + str(len(anns)))
        for n in range(len(anns)):
            x, y, w, h = anns[n]['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            # print(x, y, w, h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.imwrite(resultFile + str(image_id[i]) + 'gt.png', image)
    print("生成图片存在{}".format(resultFile))

def show_result(image_id, annFile, imageFile, resultFile):
    # image_id = 167572
    # print(image_id)
    # print(len(image_id))
    with open(annFile, "r") as f:
        row_data = json.load(f)

    image = cv2.imread(imageFile + '000000' + str(image_id) + '.jpg')
    # 读取每一条json数据
    anns = []
    for d in row_data:
        if d["image_id"] == image_id:
            x, y, w, h = d["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            anns.append([x, y, w, h])
    print("bbox number res: "+str(len(anns)))
    for n in range(len(anns)):
        x, y, w, h = anns[n]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # print(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
    cv2.imwrite(resultFile + str(image_id) + 'result.png', image)
    print("生成图片存在{}".format(resultFile))




if __name__ == "__main__":
    image_id = 306700
    # 223789 306700 329455 581781 167572
    annFile = '../maskrcnn/data/coco2017/annotations/instances_val2017.json'
    imageFile = '../maskrcnn/data/coco2017/val2017/'
    resultFile = 'visual/groundtruth/'
    showNimages(image_id, annFile, imageFile, resultFile)
    outputFile = 'det_results.json'
    outresFile = 'visual/res/'
    show_result(image_id, outputFile, imageFile, outresFile)