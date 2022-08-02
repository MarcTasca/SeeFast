import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from CamVid import CamVid
from IDDA import IDDA
from FCDiscriminator import FCDiscriminator
import torch.nn.functional as F
from torch.autograd import Variable

import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
import time
import torch.cuda.amp as amp


def val(args, model, dataloader):
    print('start val!')

    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
    return precision, miou


def train(args, curr_state, model, model_D, optimizer, optimizer_D, 
            CamVid_dataloader_train, 
            CamVid_dataloader_val,
            IDDA_dataloader_train):

    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    # implementation of amp
    scaler = amp.GradScaler()

    # loss for the model, crossentropy
    loss_func = torch.nn.CrossEntropyLoss()

    # binary cross entropy
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_miou = 0
    step = 0

    # labels for discriminator's loss
    source_label = 0
    target_label = 1
    
    # initialize curr_state
    curr_epoch = curr_state['epoch']
    curr_time = curr_state['time']
    
    #debug
    print('checkpoint step :',args.checkpoint_step)
    print('resnet :',args.context_path)
    print('epoch :', curr_epoch)
    #end

    for epoch in range(curr_epoch, args.num_epochs):

        # train the models
        model.train()
        model_D.train()

        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_D_value = 0
        loss_record = []

        # initialize learning rate
        optimizer.zero_grad()
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        optimizer_D.zero_grad()
        lr_D = poly_lr_scheduler(optimizer_D, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)

        # start counting epoch
        tic = time.perf_counter()

        # print statssss
        tq = tqdm(total=len(CamVid_dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_D %f' % (epoch, lr, lr_D))

        # idda and camvid loaders
        source_loader_iter = enumerate(IDDA_dataloader_train)
        source_size = len(IDDA_dataloader_train)
        target_loader_iter = enumerate(CamVid_dataloader_train)
        target_size = len(CamVid_dataloader_train)

        loss_total = 0

        for i in range(target_size):

            # set optimizers to zero
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # train G
            # do not accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train G with source (IDDA)
            _, batch = next(source_loader_iter)
            data, label = batch
            data = data.cuda()
            label = label.long().cuda()     
            
            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            scaler.scale(loss).backward()

            loss_seg_value += loss.data.cpu().numpy()

            # train G with target (CamVid)
            _, batch = next(target_loader_iter)
            data, _ = batch
            data = data.cuda()

            with amp.autocast():
                pred_target, _, _ = model(data)
                D_out = model_D(F.softmax(pred_target))
                loss_adv_target = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
                loss = args.lambda_adv_target * loss_adv_target

            scaler.scale(loss).backward()
            loss_adv_target_value += loss_adv_target.data.cpu().numpy()

            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train D with source (IDDA)
            pred = output.detach()

            with amp.autocast():
                D_out = model_D(F.softmax(pred)) 
                loss_D = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
                loss_D = loss_D / 2

            scaler.scale(loss_D).backward()
            loss_D_value += loss_D.data.cpu().numpy()

            # train D with target (CamVid)
            pred_target = pred_target.detach()
            with amp.autocast():
                D_out = model_D(F.softmax(pred_target))
                loss_D = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())
                loss_D = loss_D / 2
            scaler.scale(loss_D).backward()
            loss_D_value += loss_D.data.cpu().numpy()
            
            scaler.step(optimizer)
            scaler.step(optimizer_D)
            scaler.update()

            # update the stats print
            tq.update(args.batch_size)

        # this is printed at the end of every epoch
        loss_total = loss_seg_value + loss_adv_target_value * args.lambda_adv_target

        tq.set_postfix(loss='%.6f' % loss_total)
        step += 1
        writer.add_scalar('loss_step', loss_total, step)
        loss_record.append(loss_total.item())
        tq.close()

        # update curr_time
        toc = time.perf_counter() # stop timer
        curr_time += toc-tic
        print('curr_time :', curr_time)

        # print and save some stuff
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        # checkpoint : save model, discriminator and optimizers
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            path = os.path.join(args.save_model_path, 'latest_crossentropy_loss.pth')
            print('save model in %s ...' % path)
            state = {
                'epoch': epoch+1,
                'model_state': model.module.state_dict(),
                'optim_state': optimizer.state_dict(),
                'model_D_state': model_D.module.state_dict(),
                'optim_D_state': optimizer_D.state_dict(),
                'time': curr_time
            }
            torch.save(state,path)
            print('Done!')
        
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, CamVid_dataloader_val)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best_crossentropy_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)

    print('time avg :',curr_time/args.num_epochs)



def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--learning_rate_D', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--data_IDDA', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--optimizer_D', type=str, default='adam', help='optimizer discriminator , support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument("--lambda_adv_target", type=float, default=0.001, help="lambda_adv for adversarial training.")
    args = parser.parse_args(params)

    # create dataloaders camvid
    train_path = [os.path.join(args.data, 'train'), os.path.join(args.data, 'val')]
    train_label_path = [os.path.join(args.data, 'train_labels'), os.path.join(args.data, 'val_labels')]
    test_path = os.path.join(args.data, 'test')
    test_label_path = os.path.join(args.data, 'test_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')

    # loader for training
    dataset_train = CamVid(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                           loss=args.loss, mode='train')
    CamVid_dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        )

    # loader for eval
    dataset_val = CamVid(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                         loss=args.loss, mode='test')
    CamVid_dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # create dataloader idda
    train_path = os.path.join(args.data_IDDA, 'rgb')
    train_label_path = os.path.join(args.data_IDDA, 'labels')
    json_path = os.path.join(args.data_IDDA, 'classes_info.json')

    # loader for training
    dataset_train = IDDA(train_path, train_label_path, json_path, scale=(args.crop_height, args.crop_width),
                           loss=args.loss, mode='train')

    IDDA_dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        )
    
    # build model network
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    
    # build model discriminator
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if torch.cuda.is_available() and args.use_gpu:
        model_D = torch.nn.DataParallel(model_D).cuda()

    # build optimizer model
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    optimizer.zero_grad()

    # build optimizer discriminator
    optimizer_D = torch.optim.Adam(model_D.parameters(), args.learning_rate_D, betas=(0.9, 0.99), weight_decay=1e-4)
    optimizer_D.zero_grad()

    # initialize curr_state
    curr_state = {
        'epoch': 0,
        'time': 0 # seconds
    }

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        path = args.pretrained_model_path    
        print('load model from %s ...' % path)
        state = torch.load(path)

        # load bisenet and optimizer
        model.module.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optim_state'])

        # load discriminator and optimizer
        model_D.module.load_state_dict(state['model_D_state'])
        optimizer_D.load_state_dict(state['optim_D_state'])

        # load epoch and time
        curr_state['epoch'] = state['epoch']
        curr_state['time'] = state['time']
        print('Done!')

    # train
    train(args, curr_state, model, model_D, optimizer, optimizer_D, 
            CamVid_dataloader_train, 
            CamVid_dataloader_val,
            IDDA_dataloader_train)

    val(args, model, CamVid_dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate', '2.5e-2',
        '--data', './data/CamVid',
        '--num_workers', '8',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_101_sgd',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',

        # new params
        '--checkpoint_step', '20',
        '--validation_step', '5',
        '--data_IDDA', './data/IDDA',
        '--loss', 'crossentropy',
        '--optimizer_D', 'adam',
        '--learning_rate_D', '1e-4',
        # '--pretrained_model_path', './checkpoints_101_sgd/latest_crossentropy_loss.pth',
        '--lambda_adv_target', '0.001',
    ]
    main(params)