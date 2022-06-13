import torch
import torch.nn as nn
import torch.nn.functional as F


class AALS(nn.Module):
    """ Agreement-aware label smoothing """
    def __init__(self):
        super(AALS, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, logits, targets, ca):
        log_preds = self.logsoftmax(logits)  # B * C
        targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        uni = (torch.ones_like(log_preds) / log_preds.size(-1)).cuda()

        loss_ce = (- targets * log_preds).sum(1)
        loss_kld = F.kl_div(log_preds, uni, reduction='none').sum(1)
        loss = (ca * loss_ce + (1-ca) * loss_kld).mean()
        return loss


class PGLR(nn.Module):
    """ Part-guided label refinement """
    def __init__(self, lam=0.5):
        super(PGLR, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.lam = lam

    def forward(self, logits_g, logits_p, targets, ca):
        targets = torch.zeros_like(logits_g).scatter_(1, targets.unsqueeze(1), 1)
        w = torch.softmax(ca, dim=1)  # B * P
        w = torch.unsqueeze(w, 1)  # B * 1 * P
        preds_p = self.softmax(logits_p)  # B * C * P
        ensembled_preds = (preds_p * w).sum(2).detach()  # B * class_num
        refined_targets = self.lam * targets + (1-self.lam) * ensembled_preds

        log_preds_g = self.logsoftmax(logits_g)
        loss = (-refined_targets * log_preds_g).sum(1).mean()
        return loss


class InterCamProxy(nn.Module):
    """ Camera-aware proxy with inter-camera contrastive learning """
    def __init__(self, num_features, num_samples, num_hards=50, temp=0.07):
        super(InterCamProxy, self).__init__()
        self.num_features = num_features  # D
        self.num_samples = num_samples  # N
        self.num_hards = num_hards
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.temp = temp
        self.register_buffer('proxy', torch.zeros(num_samples, num_features))
        self.register_buffer('pids', torch.zeros(num_samples).long())
        self.register_buffer('cids', torch.zeros(num_samples).long())

    """ Inter-camera contrastive loss """
    def forward(self, inputs, targets, cams):
        B, D = inputs.shape
        inputs = F.normalize(inputs, dim=1).cuda()  # B * D
        sims = inputs @ self.proxy.T  # B * N
        sims /= self.temp
        temp_sims = sims.detach().clone()

        loss = torch.tensor([0.]).cuda()
        for i in range(B):
            pos_mask = (targets[i] == self.pids).float() * (cams[i] != self.cids).float()
            neg_mask = (targets[i] != self.pids).float()
            pos_idx = torch.nonzero(pos_mask > 0).squeeze(-1)
            if len(pos_idx) == 0:
                continue
            hard_neg_idx = torch.sort(temp_sims[i] + (-9999999.) * (1.-neg_mask), descending=True).indices[:self.num_hards]
            sims_i = sims[i, torch.cat([pos_idx, hard_neg_idx])]
            targets_i = torch.zeros(len(sims_i)).cuda()
            targets_i[:len(pos_idx)] = 1.0 / len(pos_idx)
            loss += - (targets_i * self.logsoftmax(sims_i)).sum()

        loss /= B
        return loss
