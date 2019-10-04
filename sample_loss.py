from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class sampleloss(nn.Module):
    def __init__(self, size_average=True):
        super(sampleloss, self).__init__()
        #self.ratio = ratio
        self.size_average = size_average

    def forward(self, ratio, inputs, targets):
        N, C = inputs.data.shape[0], inputs.data.shape[1]
        p = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1)

        probs = (p*class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        #print(ratio, type(ratio))
        batch_loss = Variable(ratio) * log_p
        #print(ratio, ratio.shape)
        #print(log_p, log_p.data.shape)
        #print(batch_loss.data.shape)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return -loss
