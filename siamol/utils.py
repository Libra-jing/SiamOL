import cv2.cv2 as cv2
import torch
import numpy as np


def convert_box2bbx(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img)
    img = img.float()
    return img


def torch_to_img(img):
    img = to_numpy(torch.squeeze(img, 0))
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


# python3
def Round(a):
    if a >= 0:
        b = 0.00000001
    else:
        b = -0.00000001
    return round(a + b)


def get_subwindow(im, pos, original_sz, avg_chans):

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2  

    context_xmin = Round(pos[0] - c)  # python3
    context_xmax = context_xmin + sz - 1
    context_ymin = Round(pos[1] - c)  # python3
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    return im_patch_original, int(context_xmin), int(context_ymin)


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2  

    context_xmin = Round(pos[0] - c)  # python3
    context_xmax = context_xmin + sz - 1
    context_ymin = Round(pos[1] - c)  # python3
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]


    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original

    return im_patch


def get_extand_searchregion(pos, sz, im, r):
    sz[0] = sz[0] * r
    sz[1] = sz[1] * r
    pred_bbox = np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])
    pred_bbox = list(map(int, pred_bbox))
    x1, x2, y1, y2 = 0, 0, 0, 0
    if pred_bbox[1] < 0:
        y1 = 0
    else:
        y1 = pred_bbox[1]
    if pred_bbox[1] + pred_bbox[3] > im.shape[0]:
        y2 = im.shape[0]
    else:
        y2 = pred_bbox[1] + pred_bbox[3]
    if pred_bbox[0] < 0:
        x1 = 0
    else:
        x1 = pred_bbox[0]
    if pred_bbox[0] + pred_bbox[2] > im.shape[1]:
        x2 = im.shape[1]
    else:
        x2 = pred_bbox[0] + pred_bbox[2]
    ext = im[y1:y2, x1:x2]
    return ext


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])  # 0-index


def rect_2_cxy_wh(rect):
    return np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]), np.array([rect[2], rect[3]])  # 0-index


def get_axis_aligned_bbox(region):
    # try:
    #     region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
    #                        region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
    # except:
    #     region = np.array(region)
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h


# anchor
def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)  # anchor num 5
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)  # anchor shape (5,4)
    size = total_stride * total_stride  # anchor size
    count = 0
    for ratio in ratios:  # height/width
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:  # scale ratio
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))  # 19*19*5=1805
    ori = - (score_size // 2) * total_stride  
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])  
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def IoU(pre_bbox, gt_bbox):
    pre_x = pre_bbox[0]
    pre_y = pre_bbox[1]
    pre_w = pre_bbox[2]
    pre_h = pre_bbox[3]

    gt_x = gt_bbox[0]
    gt_y = gt_bbox[1]
    gt_w = gt_bbox[2]
    gt_h = gt_bbox[3]

    parea = pre_w * pre_h  # pre_bbox area
    garea = gt_w * gt_h  # gt_bbox area

    x1 = max(pre_x, gt_x)
    y1 = max(pre_y, gt_y)
    x2 = min(pre_x + pre_w, gt_x + gt_w)
    y2 = min(pre_y + pre_h, gt_y + gt_h)
    w = max(0, abs(x2 - x1))
    h = max(0, abs(y2 - y1))
    area = w * h  # pre_bbox∩gt_bbox area

    iou = area / (parea + garea - area)

    return iou
