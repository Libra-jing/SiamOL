from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2.cv2 as cv2
import torch
import numpy as np
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pandas as pd
from evalyuan import evaluation

sys.path.append('..')
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from siamol.tracker import SiamOLTracker
from siamol.utils import get_axis_aligned_bbox, IoU


def main():
    parser = argparse.ArgumentParser(description='UpdateNet tracking')
    parser.add_argument('--dataset', default='OccShip', type=str, help='datasets')
    parser.add_argument('--update_path', default='../models/vot2018.pth.tar', type=str, help='update model')  # update model
    parser.add_argument('--model_path', default='../models/SiamRPNBIG.model', type=str, help='model')  # SiamRPN model
    parser.add_argument('--video', default='', type=str, help='eval one special video')  
    parser.add_argument('--Occ_degree', default='8', type=int, help='occlusion degree')  
    parser.add_argument('--vis', action='store_true', help='whether visualzie result')  
    args = parser.parse_args()
    torch.set_num_threads(5)

    dataset_root = 'E:/Sjj/' + args.dataset  # Ship/OccShip dataset root
    update_path = args.update_path
    model_path = args.model_path
    step = 3  # 1-WithoutUP  2-LinearUP  3-UpdateNet

    gpu_id = 0
    tracker = SiamOLTracker(model_path, update_path, gpu_id, step)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
                                            
    model_name = tracker.name

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        total_lost = 0
        # for v_idx, video in enumerate(dataset):
        v_idx = 0
        for video in tqdm(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            v_idx = v_idx + 1
            toc = 0
            pred_bboxes = []
            state = dict()
            print(video.name)
            for idx, (img, gt_bbox) in enumerate(video):
               # print(idx)
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    state = tracker.init(img, np.array(gt_bbox))  # 注意gt_bbox和gt_bbox_的区别
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))  # 1-based
                    pred_bbox = [cx-w/2, cy-h/2, w, h]  # 1-based
                    pred_bboxes.append(1)

                elif idx > frame_counter:
                    state = tracker.update(idx, img)
                    pos = state['target_pos']  # cx, cy
                    sz = state['target_sz']   # w, h
                    pred_bbox = np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])
                    # pred_bbox = np.array([pos[0]+1-(sz[0]-1)/2, pos[1]+1-(sz[1]-1)/2, sz[0], sz[1]])

                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
    else:
        # OPE tracking
        v_idx = 0
        for video in tqdm(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
           
            toc = 0
            v_idx = v_idx + 1
            pred_bboxes = [] 
            scores = []
            overlaps = []
            indexs = []
            track_times = []
            state = dict()
            for idx, (img, gt_bbox) in enumerate(video):   # track process
                tic = cv2.getTickCount() 
                if idx == 0:  # initial frame
                    state = tracker.init(img, np.array(gt_bbox))                      
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    pred_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                    overlap = IoU(pred_bbox, gt_bbox)
                    overlaps.append(overlap)
                    indexs.append(1)
                else:   # next frame
                    state = tracker.update(idx, img)
                    pos = state['target_pos']
                    sz = state['target_sz']
                    pred_bbox = np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])
                    pred_bboxes.append(pred_bbox)
                    overlap = IoU(pred_bbox, gt_bbox)
                    overlaps.append(overlap)
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)  # GT (green)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)  # predict bbox (yellow)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(state['score']), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
      
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, 'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path, '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:

                # Draw a score curve to see how the score changes
                
                # plt.rcParams['font.sans-serif'] = ['SimHei']
                # plt.rcParams['axes.unicode_minus'] = False
                # x = np.arange(1, len(video)+1, 1)
                # y1 = scores
                # y2 = overlaps
                # fig = plt.figure()
                # ax1 = fig.add_subplot(1, 1, 1)
                # ax1.plot(x, y1, color='blue', label='score')
                # ax1.plot(x, y2, color='green', label='overlap')
                # ax1.legend()
                # ax1.xaxis.set_ticks_position('bottom')
                # ax1.yaxis.set_ticks_position('left')
                # x_major_location = MultipleLocator(100)
                # ax1.xaxis.set_major_locator(x_major_location)
                # plt.xlabel('frame')  # 设置横坐标轴标题
                # plt.ylabel('rate')
                # fig = plt.gcf()
                # # plt.show()
                # fig.savefig(os.path.join('figs', video.name+'.png'), dpi=300)

                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('\n({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx, video.name, toc, idx / toc))
    # evaluation(args.dataset, model_name)


if __name__ == '__main__':
    main()

