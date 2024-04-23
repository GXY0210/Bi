import torch.nn as nn
import torch
import torch.autograd as autograd


def perb_att(input_att):
    gama = torch.rand(input_att.size())
    noise = gama * input_att
    att = norm(1)(noise + input_att)
    return att


class norm(nn.Module):
    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius

    def forward(self, x):
        x = self.radius * x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return x


def reconstruct_W1_loss(h_att, att):
    wt = (h_att - att).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (h_att - att).abs()
    return loss.sum() / loss.size(0)


def loss_fn(opt, recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), size_average=False)
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)


loss_2 = torch.nn.MSELoss()


def loss_fn_2(opt, recon_x, x, mean, log_var):
    mse = loss_2(recon_x, x.detach())
    mse = mse.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    loss_all = mse + KLD
    return loss_all


def generate_syn_feature(opt, netG, classes, attribute, num, return_norm=False):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            batch_att = iclass_att.repeat(num, 1)
            if opt.perb:
                batch_att = perb_att(batch_att)
            syn_att.copy_(batch_att)
            syn_noise.normal_(0, opt.tr_sigma)
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)
    if return_norm:
        out_notNorm = netG.get_out()
        return syn_feature, syn_label, out_notNorm
    else:
        return syn_feature, syn_label,


def calc_gradient_penalty(opt, netD, real_data, fake_data, input_att=None, lambda1=1):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # L_2 norm for iterpolated data
    if opt.L2_norm:
        interpolates = opt.radius * interpolates / torch.norm(interpolates, p=2, dim=1, keepdim=True)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = torch.tensor(interpolates, requires_grad=True)
    if input_att is not None:
        disc_interpolates = netD(interpolates, torch.tensor(input_att))
    else:
        disc_interpolates = netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


from __future__ import print_function
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss


# clear those instances that have no positive instances to avoid training error
# 自监督学习中的对比损失函数（Supervised Contrastive Loss）
class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        # 如果是从nn.Module继承，默认会调用方法
        self.temperature = temperature

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos * (1 - single_samples)
        loss = loss.sum() / (loss.shape[0] - single_samples.sum())

        return loss
