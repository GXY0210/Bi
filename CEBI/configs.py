# author: akshitac8
import argparse
import os
import datetime
import logging
from helper import get_logger


class OPT():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='CUB', help='FLO')
        parser.add_argument('--dataroot',
                            default='/home/staff_xiaobo_jin/prml/XinyuanRu/ZSL/Bi-VAEGAN-master/datasets/ZSL_data',
                            help='path to dataset')
        parser.add_argument('--image_embedding', default='res101')
        parser.add_argument('--class_embedding', default='att')
        parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
        parser.add_argument('--syn_num2', type=int, default=100, help='number features to generate per class')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
        parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
        parser.add_argument('--nepochBI', type=int, default=2000, help='number of epochs to train for BI')
        parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
        parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
        parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
        parser.add_argument('--classifier_lr', type=float, default=0.001,
                            help='learning rate to train softmax classifier')
        parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
        parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
        parser.add_argument('--manualSeed', type=int, help='manual seed')
        parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
        parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--gammaG_un', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--gammaD_un', type=int, default=1000, help='weight on the W-GAN loss')
        parser.add_argument('--preprocessing', action='store_true', default=False,
                            help='enbale MinMaxScaler on visual features')
        parser.add_argument('--gzslBI', action='store_true', default=False)
        ###
        parser.add_argument('--transductive', action='store_true', default=False)
        parser.add_argument('--RCritic', action='store_true', default=False, help='enable use RCritic ')
        parser.add_argument('--beta', type=float, default=1.0, help='beta for objective L_R')
        parser.add_argument('--L2_norm', action='store_true', default=False,
                            help='enbale L_2 nomarlization on visual features')
        parser.add_argument('--radius', type=int, default=1, help='radius of L_2 feature nomalization')
        parser.add_argument('--att_criterian', type=str, default='W1')
        parser.add_argument('--gammaD_att', type=float, default=10.0, help='weight on the W-GAN loss')
        parser.add_argument('--gammaG_att', type=float, default=0.1, help='weight on the W-GAN loss')
        parser.add_argument('--perb', action='store_true', default=False)
        parser.add_argument('--pretune_feature', action='store_true', default=False,
                            help='enable pre-tune visual features')
        parser.add_argument('--tune_epoch', type=int, default=15, help='pretune epochs')
        parser.add_argument('--unknown_classDistribution', action='store_true', default=False,
                            help='training in the unknown class distribution for the unseen classes')
        parser.add_argument('--no_R', action='store_true', default=False, help='no use regressor module')
        parser.add_argument('--soft', action='store_true', default=False)
        parser.add_argument('--ind_epoch', type=int, default=3, help='inductive epoch')
        parser.add_argument('--prior_estimation', type=str, default="", help='CPE or BBSE or classifier')
        parser.add_argument('--feature_type', type=str, default="", help='v or h or a or ha')

        # CE
        parser.add_argument('--matdataset', default=True, help='Data in matlab format')
        parser.add_argument('--image_embeddingCE', default='res101')
        parser.add_argument('--class_embeddingCE', default='sent', help='att or sent')
        parser.add_argument('--syn_numCE', type=int, default=100, help='number features to generate per class')
        parser.add_argument('--gzslCE', type=bool, default=True, help='enable generalized zero-shot learning')
        parser.add_argument('--preprocessing', type=bool, default=True, help='enbale MinMaxScaler on visual features')
        parser.add_argument('--standardization', action='store_true', default=False)
        parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
        parser.add_argument('--batch_sizeCE', type=int, default=2048, help='input batch size')
        parser.add_argument('--resSizeCE', type=int, default=2048, help='size of visual features')
        parser.add_argument('--attSizeCE', type=int, default=1024, help='size of semantic features')
        parser.add_argument('--nz', type=int, default=1024, help='noise for generation')
        parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
        parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')

        # network architecture

        parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
        parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
        parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

        parser.add_argument('--ins_weight', type=float, default=0.001,
                            help='weight of the classification loss when learning G')
        parser.add_argument('--cls_weight', type=float, default=0.001,
                            help='weight of the score function when learning G')
        parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
        parser.add_argument('--cls_temp', type=float, default=0.1, help='temperature in class-level supervision')

        parser.add_argument('--nepochCE', type=int, default=2000, help='number of epochs to train for CE')

        parser.add_argument('--critic_iterCE', type=int, default=5, help='critic iteration, following WGAN-GP')
        parser.add_argument('--lrCE', type=float, default=0.0001, help='learning rate to training')
        parser.add_argument('--lr_decay_epoch', type=int, default=100,
                            help='conduct learning rate decay after every 100 epochs')
        parser.add_argument('--lr_dec_rate', type=float, default=0.99, help='learning rate decay rate')
        parser.add_argument('--lambda1CE', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
        parser.add_argument('--classifier_lrCE', type=float, default=0.001,
                            help='learning rate to train softmax classifier')
        parser.add_argument('--beta1CE', type=float, default=0.5, help='beta1 for adam. default=0.5')
        parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
        parser.add_argument('--manualSeedCE', type=int, default=3483, help='manual seed')
        parser.add_argument('--nclass_allCE', type=int, default=200, help='number of all classes')
        parser.add_argument('--nclass_seenCE', type=int, default=150, help='number of all classes')  # ？

        parser.add_argument('--gpus', default='0', help='the number of the GPU to use')


        opt, _ = parser.parse_known_args()
        self.opt, self.log_dir, self.logger, self.training_logger = self.set_opt(opt)

    def return_opt(self):
        return self.opt, self.log_dir, self.logger, self.training_logger

    def set_opt(self, opt):

        opt.tag = ''
        opt.R = not opt.no_R
        opt.lambda2 = opt.lambda1
        opt.tag += f'{datetime.date.today()}_seed{opt.manualSeed}'

        if opt.unknown_classDistribution:
            opt.tag += f'_noPrior+{opt.prior_estimation}'
        if opt.pretune_feature:
            opt.tag += f'_pretuned'
        if opt.R:
            if opt.RCritic:
                opt.tag += f'_RD{opt.gammaD_att}_RG{opt.gammaG_att}'
        else:
            opt.tag += "_noR"

        if opt.transductive:
            opt.tag += f'_TransD{opt.gammaD_un}_TransG{opt.gammaG_un}'
            opt.tag += f'_RW{opt.beta}'
        if opt.L2_norm:
            opt.tag += f'_r{opt.radius}'
        log_dir = os.path.join('out', f'{opt.dataset}', f'{opt.tag}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('train')
        console = logging.StreamHandler()
        console.setLevel("INFO")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - \n%(message)s")
        console.setFormatter(formatter)
        handler = logging.FileHandler(f'{log_dir}/log.txt')
        handler.setLevel('INFO')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(f'save at {log_dir}')

        training_logger = get_logger(log_dir)
        return opt, log_dir, logger, training_logger
