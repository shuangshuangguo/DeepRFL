import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import json
from test_dataset import DataSet
from model import RFL_model
from transforms import *
from ops import ConsensusModule
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'], default='RGB')
parser.add_argument('weights', type=str)
parser.add_argument('--test_list', type=str)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
parser.add_argument('--flow_prefix', type=str, default='flow_')
args = parser.parse_args()


def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    else:
        raise ValueError("Unknown modality " + args.modality)
    with torch.no_grad():
        input_var = torch.autograd.Variable(data)
        out, _ = net(input_var)
    rst = out.data.cpu().numpy().copy()
    return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        (args.test_segments, 1, num_class)
    ), label[0]


if __name__ == '__main__':
    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'something':
        num_class = 50
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    input_mean = [104, 117, 128]
    if args.modality == 'Flow':
        input_mean = [128]
    input_std = [1]
    input_size, crop_size = 224, 224
    scale_size = input_size * 256 // 224
    net = RFL_model(num_class, 1, args.modality, init_path='', dropout=args.dropout)
    checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = checkpoint['state_dict']
    net.load_state_dict(base_dict)
    net = torch.nn.DataParallel(net, device_ids=args.gpus).cuda()
    net.eval()

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        test_list = os.path.join('splits', args.dataset, args.dataset + '_val_split_1_' + str(ratio) + '.txt')
        val_total = []
        with open(test_list, 'r') as f:
            for lines in f.readlines():
                lists = lines.strip('\n').split(' ')
                val_total.append((lists[0], int(lists[2]), ratio, int(lists[1])))  # path, label, ratio, num_frames
            f.close()
        data_loader = torch.utils.data.DataLoader(
            DataSet("", val_total, num_segments=args.test_segments,
                    new_length=data_length,
                    modality=args.modality,
                    image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB',
                                                                     'RGBDiff'] else args.flow_prefix + "{}_{:05d}.jpg",
                    test_mode=True,
                    transform=torchvision.transforms.Compose([
                        cropping,
                        Stack(roll=args.arch == 'BNInception'),
                        ToTorchFormatTensor(div=args.arch != 'BNInception'),
                        GroupNormalize(input_mean, input_std),
                    ])),
            batch_size=10, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        data_gen = enumerate(data_loader)
        total_num = len(data_loader.dataset)
        output = []

        proc_start_time = time.time()
        max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

        for i, (data, label) in data_gen:
            if i >= max_num:
                break
            rst = eval_video((i, data, label))
            output.append(rst[1:])
            cnt_time = time.time() - proc_start_time
            if i % 500 == 0:
                print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                                total_num,
                                                                                float(cnt_time) / (i + 1)))
        video_out = [np.mean(x[0], axis=0) for x in output]
        video_pred = [np.argmax(x) for x in video_out]
        video_labels = [x[1] for x in output]
        cf = confusion_matrix(video_labels, video_pred).astype(float)
        cf_path = os.path.join(os.getcwd(), 'scores', args.dataset, '_'.join((args.modality.lower(), 'score')),
                               '_'.join((str(checkpoint['best_prec1']), args.modality, str(ratio))) + '_25s10c_cf')
        np.savez(cf_path, cf)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        print(cls_acc)
        print('for ratio {}, Accuracy {:.02f}%'.format(ratio, np.mean(cls_acc) * 100))

        if args.save_scores is not None:
            save_scores = args.save_scores
        else:
            save_scores = os.path.join(os.getcwd(), 'scores', args.dataset, '_'.join((args.modality.lower(), 'score')),
                                       '_'.join(
                                           (str(checkpoint['best_prec1']), args.modality, str(ratio))) + '_25s10c.npz')
        # reorder before saving
        name_list = [x[0] for x in val_total]
        order_dict = {e: i for i, e in enumerate(sorted(name_list))}
        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)
        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]
        np.savez(save_scores, scores=reorder_output, labels=reorder_label)
