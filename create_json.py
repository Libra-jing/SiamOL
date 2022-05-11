import json
import os
import math


def fun(str_num):
    before_e = float(str_num.split('e')[0])
    sign = str_num.split('e')[1][:1]
    after_e = int(str_num.split('e')[1][1:])

    if sign == '+':
        float_num = before_e * math.pow(10, after_e)
    elif sign == '-':
        float_num = before_e * math.pow(10, -after_e)
    else:
        float_num = None
        print('error: unknown sign')
    return int(round(float_num))


def create_json():
    shipdata_path = "E:/Sjj/SiamOL/datasets/OccShip"
    parents = os.listdir(shipdata_path)
    json_arr = []
    attr = ["Illumination Variation", "Out-of-Plane Rotation", "Scale Variation", "Occlusion", "Deformation", "Motion Blur", "Fast Motion", "In-Plane Rotation", "Out-of-View", "Background Clutters", "Low Resolution"]
    for each_video in parents:
        
        # read groundtruth_rect
        gt_path = os.path.join(shipdata_path, each_video, "groundtruth_rect.txt")
        with open(gt_path, 'r') as f:  # open
            lines = f.readlines()  # read all lines
            res_gt = []
            for i, line in enumerate(lines):
                if 'e' in line:
                    tmp = list(map(fun, line.split()))  
                else:
                    tmp = list(map(int, line.split()))
                res_gt.append(tmp)
            first_line = res_gt[0]  # first line

        # read img_name
        img_path = os.path.join(shipdata_path, each_video, "img")
        res_img = []
        img_list = os.listdir(img_path)
        num = 0
        for each_img in img_list:
            if each_img.split('.')[1] == 'jpg':
                num = num + 1
                res_img.append(os.path.join(each_video, "img", each_img).replace('\\', '/'))

        if(len(lines) == num): 
            print(each_video, ": True")
        else:
            print(each_video, ": False")
        
        # read video attributes
        att_path = "E:/Sjj/SiamOL/datasets/OccShip"
        res_att = []
        with open(os.path.join(att_path, each_video, "att.txt"), 'r') as f:  
            line = f.readlines()[0]  
            if 'e' in line:
                att = list(map(fun, line.split())) 
            else:
                att = list(map(int, line.split()))
        for index, item in enumerate(att):
            if item == 1:
                res_att.append(attr[index])

        one = {"video_dir": each_video, "init_rect": first_line, "img_names": res_img,  "gt_rect": res_gt, "attr": res_att}
        two = {each_video: one}
        json_arr.append(two)
    with open('OccShip.json', 'w') as outfile:
        json.dump(json_arr, outfile)


if __name__ == '__main__':
    create_json()


