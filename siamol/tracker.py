import heapq
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt

from momentum import Momentum
from .utils import get_subwindow_tracking, generate_anchor, get_axis_aligned_bbox, Round, APCE, im_to_torch, \
    get_subwindow
from .net import SiamRPNBIG
from .config import config
from updatenet.net_upd import UpdateResNet512, UpdateResNet256
from .classifier.base_classifier import BaseClassifier
from .classifier.config import cfg
from .classifier.anchor import Anchors
import cv2


class SiamOLTracker:

    def __init__(self, model_path, update_path, gpu_id, step=1):

        self.gpu_id = gpu_id
        self.net = SiamRPNBIG()  
        self.is_deterministic = False  

        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            self.net.load_state_dict(torch.load(model_path)['model'])  #
        else:
            self.net.load_state_dict(torch.load(model_path))
        with torch.cuda.device(self.gpu_id):
            self.net = self.net.cuda()
        self.net.eval()
        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier = BaseClassifier(self.net)  # off-line classifier

        self.state = dict()

        self.step = step  # 1,2,3
        if self.step == 1:
            self.name = 'WithoutUP'
        elif self.step == 2:
            self.name = 'LinearUP'
        else:
            dataset = update_path.split('/')[-1].split('.')[0]
            if dataset == 'vot2018' or dataset == 'vot2016':
                self.name = 'Our'
            else:
                self.name = dataset

        if self.step == 3:
            # load UpdateNet network
            self.updatenet = UpdateResNet512()
            update_model = torch.load(update_path)['state_dict']

            update_model_fix = dict()
            for i in update_model.keys():
                if i.split('.')[0] == 'module':  
                    update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
                else:
                    update_model_fix[i] = update_model[i]  

            self.updatenet.load_state_dict(update_model_fix)
            self.updatenet.eval().cuda()
        else:
            self.updatenet = ''

    def tracker_eval(self, x_crop, target_pos, target_sz, scale_z, p):
        with torch.no_grad():  # avoid out of memory
            delta, score = self.net(x_crop)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

        delta[0, :] = delta[0, :] * self.anchors[:, 2] + self.anchors[:, 0]
        delta[1, :] = delta[1, :] * self.anchors[:, 3] + self.anchors[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * self.anchors[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * self.anchors[:, 3]

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        def normalize(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2

            hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep

        # size penalty
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1.) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        if cfg.TRACK.USE_CLASSIFIER:
            flag, s = self.classifier.track()   # off-line classifier

            confidence = Image.fromarray(s.detach().cpu().numpy())
            confidence = np.array(confidence.resize((self.score_size, self.score_size))).flatten()
            ss = pscore.reshape(5, -1)
            # score fusion
            pscore = pscore.reshape(5, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                     normalize(confidence) * cfg.TRACK.COEE_CLASS

            x = int(confidence.shape[0] ** 0.5)   

            pscore = pscore.flatten()

        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE

        
        pscore_1 = torch.from_numpy(pscore).view(1, 5, x, x).cpu().detach().numpy().sum(axis=1)
        pscore_1 = pscore_1.astype(np.float32)
        c = _nms(torch.from_numpy(pscore_1).view(1, 1, x, x)).numpy()


        best_idx = np.argmax(pscore)
        worst_idx = np.argmin(pscore)
        bbox = delta[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR


        cx, cy = bbox[0] + target_pos[0], bbox[1] + target_pos[1]
        width = target_sz[0] * (1 - lr) + bbox[2] * lr
        height = target_sz[1] * (1 - lr) + bbox[3] * lr

        best_score = score[best_idx]
        worst_score = score[worst_idx]

        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier.update(bbox, scale_z, flag) # off-line classifier update

        target_pos = np.array([cx, cy])
        target_sz = np.array([width, height])
        return target_pos, target_sz, best_score, worst_score, score, c

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def init(self, im, init_rbox):

        state = self.state

        [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
        # tracker init
        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
        self.saves = []
        self.saves.append([target_pos, target_sz, im, 1])

        p = config
        p.update(self.net.cfg)

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        if p.adaptive:
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = 287  # small object big search region
            else:
                p.instance_size = 271
            # python3
            p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1  # python
            self.score_size = (p.instance_size - p.exemplar_size) // \
                              p.total_stride + 1
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.lost_count = 0

        p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = Round(np.sqrt(wc_z * hc_z))  # python3

        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = im_to_torch(z_crop)
        z_crop = Variable(z_crop.unsqueeze(0))

        if self.step == 1:  # not update
            self.net.temple(z_crop.cuda())  # initial template
        else:  # update
            z_f = self.net.featextract(z_crop.cuda())  # [1,512,6,6]
            self.net.kernel(z_f)
            state['z_f'] = z_f.cpu().data  # accumulated template
            state['z_0'] = z_f.cpu().data  # initial template

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.tile(window.flatten(), p.anchor_num)

        if cfg.TRACK.USE_CLASSIFIER:  # off-line classifier initialization
            s_xx = s_z * (p.instance_size * 2 / p.exemplar_size)
            x_crop = Variable(im_to_torch(get_subwindow_tracking(im, target_pos, p.instance_size * 2, round(s_xx), avg_chans)).unsqueeze(0))
            self.classifier.initialize(x_crop.type(torch.FloatTensor), init_rbox, p)

        state['p'] = p
        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['max_score'] = 1
        state['next_dect'] = 0
        self.state = state

        return state

    def update(self, idx, im):
        tmp = 0
        clone = im.copy()
        state = self.state
        p = state['p']
        avg_chans = state['avg_chans']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        cur_dect = state['next_dect'] # current tracking state
        th = 0.93  # th
        tl = 0.92  # tl
        tol = 0.02  # tol

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)  #

        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # python3
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_c = get_subwindow_tracking(im, target_pos, p.instance_size, Round(s_x), avg_chans)
        x_c = im_to_torch(x_c)
        x_crop = Variable(x_c.unsqueeze(0))
        target_pos, target_sz, max_score, min_score, score, c = self.tracker_evalcf(x_crop.cuda(), target_pos, target_sz, scale_z, p)
        c_tmp = c.flatten()
        c_2nd_max = np.sort(c_tmp)[-2]

        next_dect = -1
        # Occlusion determination mechanism:
        if (max_score > 0 and (max_score - state['max_score']) >= 0 and cur_dect == 0):
            next_dect = 0
            tmp = 1
        if (max_score >= th and (max_score - state['max_score']) >= 0 and cur_dect == 1):
            next_dect = 0
            tmp = 2
        if (max_score > 0) and (max_score - state['max_score']) < 0 and (cur_dect == 0) and abs(max_score - state['max_score']) < tol:
            next_dect = 0
            tmp = 3
        if (max_score <= 0) and (max_score - state['max_score']) >= 0 and (cur_dect == 0):
            next_dect = 0
            tmp = 4
        if (max_score < th and max_score > 0) and (max_score - state['max_score']) >= 0 and cur_dect == 1:
            next_dect = 1
            tmp = 5
        if (max_score > 0 and (max_score - state['max_score']) < 0 and cur_dect == 1) or (max_score > 0 and (max_score - state['max_score']) < 0 and abs(max_score - state['max_score']) >= tol):
            next_dect = 1
            tmp = 6
        if max_score < tl and (max_score - state['max_score']) >= 0 and cur_dect == 1:
            next_dect = 1
            tmp = 7
        if tmp == 0:
            os._exit(0)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

        print(idx, max_score)
        s = []
        if next_dect == 0:
            self.saves.append([target_pos, target_sz, im, max_score])  # template pool
            # update process
            if self.step > 1:
                z_crop = Variable(im_to_torch(get_subwindow_tracking(im, target_pos, p.exemplar_size, Round(s_z), avg_chans)).unsqueeze(0))
                z_f = self.net.featextract(z_crop.cuda())  # detection template
                if self.step == 2:  # 1-LinearUP
                    zLR = 0.0102  
                    z_f_ = (1 - zLR) * Variable(state['z_f']).cuda() + zLR * z_f  # accumulated template
                else:  # 2-Our
                    temp = torch.cat((Variable(state['z_0']).cuda(), Variable(state['z_f']).cuda(), z_f), 1)
                    init_inp = Variable(state['z_0']).cuda()
                    z_f_ = self.updatenet(temp, init_inp)

                state['z_f'] = z_f_.cpu().data  # accumulated template
                self.net.kernel(z_f_)  # update
        if tmp == 2 or cur_dect == 1:  # deal with occlusion
            for x in self.saves:
                s.append(x[3])
            max_index = map(s.index, heapq.nlargest(1, s))
            y = list(max_index)
            pos = []
            sz = []
            image = []
            for item, x in enumerate(self.saves):
                if item == y[0]:
                    pos = x[0]
                    sz = x[1]
                    image = x[2]
                    break
            print(pos, sz)
            pred_bbox = np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])
            pred_bbox = list(map(int, pred_bbox))
            template = image[pred_bbox[1]:pred_bbox[1] +pred_bbox[3], pred_bbox[0]:pred_bbox[0] +pred_bbox[2]]
            target_pos, target_sz = self.show_window(im, template, target_pos, target_sz)

        pred_bbox = np.array([target_pos[0] - target_sz[0] / 2, target_pos[1] - target_sz[1] / 2, target_sz[0], target_sz[1]])
        pred_bbox = list(map(int, pred_bbox))
        cv2.rectangle(clone, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)  
        cv2.putText(clone, str(idx + 1), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(clone, str(state['max_score']), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['max_score'] = max_score
        state['next_dect'] = next_dect  
        self.state = state
        return state

    # generating the sliding window
    def sliding_window(self, image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    # displaying the image on which the sliding window is displayed
    def show_window(self, im, template, pos, sz):  
        wind_row = template.shape[1]*5
        wind_col = template.shape[0]*5
        step = (template.shape[1] + template.shape[0])//2
        count = 0
        pre_pos = []
        pre_sz = []
        coords = []
        regions = []
        for (x, y, window) in self.sliding_window(im, step, (wind_row, wind_col)):
            count = count + 1
            if window.shape[0] != wind_col or window.shape[1] != wind_row:
                continue
          
            candi_region = im[y:y + wind_col, x:x + wind_row]  # the image which has to be predicted
            meth = 'cv2.TM_SQDIFF_NORMED'
            img = candi_region.copy()
            method = eval(meth)
            template_threshold = 0.50

            dets = nms(img, template, template_threshold)
            print(dets.tolist())
            list_dets = dets.tolist()
            if list_dets != []:
                coord = list_dets[0]
                coords.append(coord[4])
                regions.append(candi_region)
                cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)
                cv2.putText(img, str(coord[4]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cx, cy, w, h = get_axis_aligned_bbox(np.array([x+int(coord[0]), y+int(coord[1]), int(coord[2]-coord[0]), int(coord[3]-coord[1])]))
                pre_pos, pre_sz = np.array([cx, cy]), np.array([w, h])
                
        if pre_pos == []:
            pre_pos = pos
            pre_sz = sz

        return pre_pos, pre_sz

   
def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# eliminate redundant boxes
def nms(img_gray, template_img, template_threshold):
    import time
    h, w = template_img.shape[:2]
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    start_time = time.time()
    loc = np.where(res >= template_threshold) 
    score = res[res >= template_threshold]  
   
    xmin = np.array(loc[1])
    ymin = np.array(loc[0])
    xmax = xmin + w
    ymax = ymin + h
    xmin = xmin.reshape(-1, 1) 
    xmax = xmax.reshape(-1, 1)  
    ymax = ymax.reshape(-1, 1) 
    ymin = ymin.reshape(-1, 1)  
    score = score.reshape(-1, 1) 
    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(data_hlist)  
    thresh = 0.3  

    keep_dets = py_nms(data_hstack, thresh)
    print("nms time:", time.time() - start_time)  
    dets = data_hstack[keep_dets]  
    return dets



