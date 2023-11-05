import torch
from collections import OrderedDict

def filter_params(ckpt, left_prefix, inverse=False):
    state_dict = ckpt['state_dict']
    filter_state_dict = OrderedDict()

    if not inverse:     # default
        for key, value in state_dict.items():
            for prefix in left_prefix:
                if prefix in key:
                    filter_state_dict[key] = value
    else:
        for key, value in state_dict.items():
            for prefix in left_prefix:
                if prefix not in key:
                    filter_state_dict[key] = value

    ckpt['state_dict'] = filter_state_dict
    return ckpt

if __name__ == '__main__':
    ckpt_names = [
        'work_dirs/stream_petrv2_seq_v2_eva1k/iter_66048.pth'
    ]
    left_prefix = [
        'img_backbone',
    ]
    # left_prefix = left_prefix + [
    #     'pts_bbox_head'
    # ]



    for ckpt_name in ckpt_names:
        ckpt = torch.load(ckpt_name)
        filter_ckpt = filter_params(ckpt, left_prefix)

        save_name = ckpt_name[:-4] + '_backbone' + '.pth'
        torch.save(filter_ckpt, save_name)
