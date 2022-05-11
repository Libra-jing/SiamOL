import numpy as np


class Config:
    # These are the default hyper-params
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    template_img_size = 127
    detection_img_size = 271
    total_stride = 8
    score_size = int((instance_size-exemplar_size)/total_stride+1)
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    
    # adaptive change search region #
    adaptive = True

    out_scale = 0.001
    response_sz = 17
    response_up = 16

    size = anchor_num * score_size * score_size
    coee_class = 0.8

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)   # self.k = v
        # python3 //
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1

config = Config()