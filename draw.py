# encoding: utf-8
import os
import numpy as np
import cv2

from toolkit.datasets import DatasetFactory

src_pic_path = 'E:/Sjj/SiamOL/datasets/OccShip'

all_pic = os.listdir(src_pic_path)  
all_pic.sort()
data = DatasetFactory.create_dataset(name='OccShip', dataset_root=src_pic_path, load_img=False)

each_video_path = [os.path.join(src_pic_path, video_path) for video_path in all_pic]

for index, item in enumerate(each_video_path):
    one_video = each_video_path[index]
    result = one_video.split('\\')
    video_name = result[len(result) - 1]
    print('choosed video_name: ({})--'.format(index), video_name)

    
    eval_result_path = './bin/results/OccShip'
    all_eval_trackers = os.listdir(eval_result_path)
    all_eval_trackers.sort()
    each_eval_tracker = [os.path.join(eval_result_path, eval_tracker) for eval_tracker in all_eval_trackers]  # evaluation results of all trackers
    each_eval_tracker.sort()

    # choose one tracker
    one_tracker = each_eval_tracker[1]  
    each_eval_result = os.listdir(one_tracker)
    each_eval_result.sort()  
    # ---------------------------------------------
    all_trakcers_result = []
    for tracker in each_eval_tracker:
        each_eval_result_path = [os.path.join(tracker, eval_result) for eval_result in each_eval_result]
        all_trakcers_result.append(each_eval_result_path)
# ---------------------------------------------

    all_list = []  # a video of one tracker
    for num in range(0, len(all_trakcers_result)):
        with open(all_trakcers_result[num][index]) as eval_result:  # choose a video of one tracker
            dataset = []
            lines = eval_result.readlines()

            # read datas in txt file, transform to String formation
            for line in lines:
                temp1 = line.strip('\n')
                temp2 = temp1.split('\t')
                dataset.append(temp2)

            import re

            new_dataset = [re.split(',', new_line[0].strip()) for new_line in dataset]  
            
            for i in range(0, len(new_dataset)):
                for j in range(len(new_dataset[i])):
                    new_dataset[i][j] = int(float(new_dataset[i][j]))
            all_list.append(new_dataset)

    
    # every frame in a video
    frames_list = os.listdir(os.path.join(one_video, 'img'))
    frames_list.sort()
    frames_path = [os.path.join(os.path.join(one_video, 'img'), frame_path) for frame_path in frames_list]

    

    # draw
    dst_pic_path = 'E:/Sjj/SiamOL/visualResults/'
    dst_pic_path = dst_pic_path + video_name
    f = os.path.exists(dst_pic_path)
    if f is True:
        continue
    else:
        os.makedirs(dst_pic_path)
        # show the tracking results
    for index, path in enumerate(frames_path):
            spath = path.split('\\')
            if (spath[len(spath)-1].split('.')[1]) != 'jpg':
                continue
            img = cv2.imread(path)
           
            track_gt = all_list[1][index]  # GT
           
           # draw bounding boxes
            cv2.rectangle(img, (track_gt[0], track_gt[1]), (track_gt[0] + track_gt[2], track_gt[1] + track_gt[3]), (0, 255, 0), thickness=2)  

            track_gt_1 = all_list[0][index]  # tracker1
            cv2.rectangle(img, (track_gt_1[0], track_gt_1[1]), (track_gt_1[0] + track_gt_1[2], track_gt_1[1] + track_gt_1[3]), (0, 0, 255), thickness=2) 

            track_gt_2 = all_list[2][index]  # tracker2
            cv2.rectangle(img, (track_gt_2[0], track_gt_2[1]), (track_gt_2[0] + track_gt_2[2], track_gt_2[1] + track_gt_2[3]), (255, 0, 0), thickness=2)  

            track_gt_3 = all_list[3][index]  # tracker3
            cv2.rectangle(img, (track_gt_3[0], track_gt_3[1]), (track_gt_3[0] + track_gt_3[2], track_gt_3[1] + track_gt_3[3]), (255, 0, 255), thickness=2)  

            cv2.putText(img, '#{}'.format(index+1), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, video_name, (10, int(img.shape[0] * 0.9)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.imwrite(dst_pic_path + '/' + video_name + '_img_{}.jpg'.format(index+1), img)
           

