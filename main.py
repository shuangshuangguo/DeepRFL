import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from dataset import DataSet
from model import RFL_model
from sample_loss import sampleloss
from transforms import *
from opts import parser
import numpy as np
import torch.optim.lr_scheduler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(1)  # cpu
torch.cuda.manual_seed_all(1)  # gpu
np.random.seed(1)  # numpy
random.seed(1)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.modality == 'RGB':
        data_length = 1
    elif args.modality == 'Flow':
        data_length = 5
    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'something':
        num_class = 50
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    print(args)

    model = RFL_model(num_class, args.num_segments, args.modality, init_path=args.finetune, dropout=args.dropout)
    params = model.get_optim_policies()
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading from checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    crop_size = model.input_size
    scale_size = model.input_size * 256 // 224
    input_mean = model.input_mean
    input_std = model.input_std

    train_augmentation = model.get_augmentation()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    cudnn.benchmark = True
    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)
    train_loader = torch.utils.data.DataLoader(
        DataSet(args.train_list, num_segments=args.num_segments, new_length=data_length, \
                modality=args.modality,
                image_tmpl="img_{:05d}.jpg" if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
                transform=torchvision.transforms.Compose([train_augmentation,
                                                          Stack(roll=True), \
                                                          ToTorchFormatTensor(div=False), normalize, ])), \
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        DataSet(args.val_list, num_segments=args.num_segments, new_length=data_length, \
                modality=args.modality,
                image_tmpl="img_{:05d}.jpg" if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
                random_shift=False, transform=torchvision.transforms.Compose([
                GroupScale(int(scale_size)), GroupCenterCrop(crop_size), \
                Stack(roll=True),
                ToTorchFormatTensor(div=False),
                normalize, ])), batch_size=args.batch_size, shuffle=False, \
        num_workers=args.workers, pin_memory=True)

    cls_criterion = torch.nn.CrossEntropyLoss().cuda()
    samp_criterion = sampleloss().cuda()
    sim_criterion = torch.nn.MSELoss().cuda()

    optim = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optim, epoch, args.lr_steps)
        # train for one epoch
        train(train_loader, model, cls_criterion, sim_criterion, optim, epoch)

        # evaluate on validation set
        # eval_freq is the controller of when to save and validate the model
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss, prec1 = validate(val_loader, model, cls_criterion)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
            print('best_prec1: ', best_prec1)


def train(train_loader, model, cls_criterion, sim_criterion, optim, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    '''calculate similarity losses between part and full features'''
    sim_losses = AverageMeter()
    '''calculate prediction losses of part and full features'''
    cls_losses = AverageMeter()
    full_losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for m, (full_input, target, _, part_input, part_target, ratio) in enumerate(train_loader):
        assert np.array_equal(target.numpy(),
                              part_target.numpy()), "part video and complete video should have same label"
        data_time.update(time.time() - end)
        with torch.no_grad():
            part_input = part_input.cuda()
            part_target = part_target.cuda()
            part_input_var = torch.autograd.Variable(part_input)
            part_target_var = torch.autograd.Variable(part_target)
        part_out, feature_part = model(part_input_var, part=True)
        with torch.no_grad():
            full_input = full_input.cuda()
            full_input_var = torch.autograd.Variable(full_input)
        full_out, feature_full = model(full_input_var, part=False)
        feature_full = torch.autograd.Variable(feature_full.data, requires_grad=False)

        loss_cls = cls_criterion(part_out, part_target_var)
        # ratio = ratio.type(torch.cuda.FloatTensor)
        # loss_cls = samp_criterion(ratio, part_out, part_target_var)
        loss_full = cls_criterion(full_out, part_target_var)
        loss_sim = sim_criterion(feature_part, feature_full)
        loss = loss_cls + loss_full + loss_sim

        optim.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
        optim.step()

        cls_losses.update(loss_cls.item(), full_input.size(0))
        sim_losses.update(loss_sim.item(), full_input.size(0))
        prec1, _ = accuracy(part_out.data, part_target, topk=(1, 5))
        top1.update(prec1.item(), part_input.size(0))
        full_losses.update(loss_full.item(), full_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if m % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'cls_loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'full_loss {full_loss.val:.4f} ({full_loss.avg:.4f})  '
                  'sim_loss {sim_loss.val:.4f} ({sim_loss.avg:.4f})  '
                  'prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, m, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=cls_losses, full_loss=full_losses, sim_loss=sim_losses,
                top1=top1, lr=optim.param_groups[-1]['lr']))


def validate(val_loader, model, cls_criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    for i, (input, target, ratio) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        # compute output
        output, feature = model(input_var, part=True)
        loss = cls_criterion(output, target_var)
        losses.update(loss.item(), input.size(0))
        prec1, _ = accuracy(output.data, target, topk=(1, 5))
        # update the sum of validation examples with input.size(0)
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader), loss=losses, top1=top1))

    print('loss: ', losses.avg)
    print('accuracy: ', top1.avg)
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    dirname = os.path.join('models', args.snapshot_pref)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filename = '_'.join((args.snapshot_pref, str(state['epoch']), filename))
    filename = os.path.join(dirname, filename)
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join(
            (args.snapshot_pref, 'model_best_', str(state['epoch']), str(state['best_prec1']) + '.pth.tar'))
        best_name = os.path.join('models', args.snapshot_pref, best_name)
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    '''update all value of the object'''

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
