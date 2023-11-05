import torch
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset
import os
import importlib
import torchvision
import numpy as np
import cv2
from PIL import ImageDraw
from PIL import Image
import sys
from mmdet3d.models import build_model
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core import MlvlPointGenerator

flag_use_server = True
if flag_use_server:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _bbox_decode(priors, bbox_preds):
    xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
    whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
    tl_x = (xys[..., 0] - whs[..., 0] / 2)
    tl_y = (xys[..., 1] - whs[..., 1] / 2)
    br_x = (xys[..., 0] + whs[..., 0] / 2)
    br_y = (xys[..., 1] + whs[..., 1] / 2)
    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes

def _centers2d_decode(priors, pred_centers2d):
    centers2d = (pred_centers2d[..., :2] * priors[:, 2:]) + priors[:, :2]
    return centers2d

class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
mean=[0.406, 0.456, 0.485]
# std=[1/255, 1/255, 1/255]
std=[58.395/255, 57.12/255, 57.375/255]
denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=mean, std=std),
            torchvision.transforms.ToPILImage(),
        ))

# A. Config and Path    # TODO set
configs = ['projects/configs/0426/stream_petrv2_yolo2dp_top50.py',
           'projects/configs/0426/stream_petrv2_yolo2dp_th1e-1.py',
           'projects/configs/0506/stream_petrv2_yolo2dp_th1e-1.py',
           'projects/configs/0506/stream_petrv2_yolo2dp_th1e-2.py',
            'projects/configs/a100/stream_petrv2_seq_v2_eva.py',
           ]
ckpts = ['work_dirs/stream_petrv2_yolo2dp_top50/latest.pth',
         'work_dirs/stream_petrv2_yolo2dp_th1e-1/latest.pth',
         'work_dirs/stream_petrv2_yolo2dp_th1e-1/latest.pth',
         'work_dirs/stream_petrv2_yolo2dp_th1e-2/latest.pth',
         'work_dirs/stream_petrv2_seq_v2_eva/latest.pth',
         ]
prefixs = ['top50_', 'th1e1_', 'th1e1e2_', 'th1e2e2_', 'eva2048_']
kth = 4
config = configs[kth] #
arg_checkpoint = ckpts[kth] #
save_prefix = prefixs[kth] #
threshold_score = 0.01  # TODO set
save_prefix = save_prefix + '{:.2f}_'.format(threshold_score)

cfg = Config.fromfile(config)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
plugin_dir = cfg.plugin_dir
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]
for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
sys.path.append(os.getcwd())
plg_lib = importlib.import_module(_module_path)

# B. Dataset and dataloader
indexs = [0, 300, 600, 900, 1200]   # [0, 900]  # TODO set

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# B1. build the model and load checkpoint
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
load_checkpoint(model, arg_checkpoint, map_location='cpu')
model.CLASSES = dataset.CLASSES
memory = []
hook = [model.img_roi_head.register_forward_hook(lambda self, input, output: memory.append(output))]    # frustum

# B2. Forward test process
model = MMDataParallel(model, device_ids=[0])
model.eval()
# loader = iter(data_loader)
# for i in range (ith+1):
#     data = next(loader)
for ith, data in enumerate(data_loader):
    if ith > indexs[-1]:
        break
    if ith not in indexs:
        continue
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **data)

    # B3. obtain out keys
    preds_dicts = memory[-1]
    prior_generator = MlvlPointGenerator([16,], offset=0)   # TODO rollback [8, 16, 32]
    cls_scores = preds_dicts['enc_cls_scores']      # level x (BN, C, H, W)
    bbox_preds = preds_dicts['enc_bbox_preds']
    pred_centers2d_offset = preds_dicts['pred_centers2d_offset']
    objectnesses = preds_dicts['objectnesses']
    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    mlvl_priors = prior_generator.grid_priors(featmap_sizes,  dtype=cls_scores[0].dtype,
        device=cls_scores[0].device, with_stride=True)
    num_imgs = preds_dicts['enc_cls_scores'][0].shape[0]
    flatten_cls_scores = [                           # level x (BN, HW, C)
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, 26)
        for cls_score in cls_scores
    ]
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
        for objectness in objectnesses]

    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds]
    flatten_centers2d_preds = [
        pred_center2d_offset.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
        for pred_center2d_offset in pred_centers2d_offset]
    valid_indices_list = []
    for jth in range(len(cls_scores)):
        # sample_weight = cls_scores[jth].topk(1, dim=1).values.sigmoid()  # (BN, 1, Hi, Wi)  # cls_score as sample weight
        sample_weight = objectnesses[jth].sigmoid() * cls_scores[jth].topk(1, dim=1).values.sigmoid() # cls_score * obj
        # sample_weight = objectnesses[jth].sigmoid()
        sample_weight_nms = torch.nn.functional.max_pool2d(sample_weight, (3, 3), stride=1, padding=1)
        sample_weight_nms = sample_weight_nms.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)  # (BN, Hi*Wi, 1)
        sample_weight_ = sample_weight.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
        sample_weight = sample_weight_ * (sample_weight_ == sample_weight_nms).float()  # (BN, Hi*Wi, 1)
        valid_indices = sample_weight > threshold_score
        valid_indices_list.append(valid_indices)
        ## obj vers.
        # sample_weight = obj.sigmoid()
        # sample_weight_nms = torch.nn.functional.max_pool2d(sample_weight, (3, 3), stride=1, padding=1)
        # sample_weight_nms = sample_weight_nms.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
        # sample_weight_ = sample_weight.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
        # sample_weight = sample_weight_ * (sample_weight_ == sample_weight_nms).float()
        # valid_indices = sample_weight > threshold_score
        # valid_indices_list.append(valid_indices)
    valid_indices = torch.cat(valid_indices_list, dim=1)
    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid().max(dim=-1, keepdim=True).values  # (BN, sum(Hi*Wi), num_cls)
    flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_centers2d_preds = torch.cat(flatten_centers2d_preds, dim=1)
    flatten_priors = torch.cat(mlvl_priors)
    flatten_bboxes = _bbox_decode(flatten_priors, flatten_bbox_preds)
    flatten_centers2d = _centers2d_decode(flatten_priors, flatten_centers2d_preds)
    pred_bbox_list = []
    pred_centers2d_list = []
    pred_cls_score_list = []
    pred_obj_list = []
    for i in range(num_imgs):
        pred_bbox = flatten_bboxes[i][valid_indices[i].repeat(1, 4)].reshape(-1, 4)
        pred_centers2d = flatten_centers2d[i][valid_indices[i].repeat(1, 2)].reshape(-1, 2)
        pred_cls_score = flatten_cls_scores[i][valid_indices[i].repeat(1, 1)].reshape(-1, 1)
        pred_obj = flatten_objectness[i][valid_indices[i].repeat(1, 1)].reshape(-1, 1)
        # dets, _ = self._bboxes_nms(cls_scores, bboxes, score_factor, cfg)
        # bbox = bbox_xyxy_to_cxcywh(pred_bbox)
        # print(bbox)
        pred_bbox_list.append(pred_bbox)
        pred_centers2d_list.append(pred_centers2d)
        pred_cls_score_list.append(pred_cls_score)
        pred_obj_list.append(pred_obj)

    # B4. Plot images and labels
    plt.figure(figsize=(80, 24))    # (40, 12)
    for i in range (num_imgs):
        plt.subplot(2, 4, i + 1)
        img = denormalize_img(data['img'][0].data[0][0][i])
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        pic = ImageDraw.ImageDraw(img)
        bboxes = pred_bbox_list[i]
        centers2d = pred_centers2d_list[i]
        cls_scores = pred_cls_score_list[i]
        objs = pred_obj_list[i]
        for bbox in bboxes:
            pic.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]),fill=None,outline ='blue',width =3)  # 5
        for center2d, score, obj in zip(centers2d, cls_scores, objs):
            pic.ellipse((center2d[0]-3, center2d[1]-3, center2d[0]+3, center2d[1]+3), fill='yellow',)
            pic.text((center2d[0]-3, center2d[1]-3), '{:.2f}'.format(score[0].item()*obj[0].item()))
        # plot write how many boxes in current frame
        if len(cls_scores) > 0:
            _weights = cls_scores * objs #
            _wmax, _wmin = _weights.max().item(), _weights.min().item()
            plt.title("Proposal num: %d, max: %.2f, min: %.2f" % (len(cls_scores), _wmax, _wmin))
        plt.imshow(img)


    if flag_use_server:
        plt.savefig('vis/vis_2dbox/%s_box2d_%d.png' % (save_prefix, ith))
    else:
        plt.show()
    plt.clf()

