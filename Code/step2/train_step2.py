import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.CamVid import CamVid
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
import time
import torch.cuda.amp as amp

def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

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

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def train(args, model, optimizer, curr_state, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss()

    max_miou = 0
    step = 0
    
    # new 2
    # in order to retrieve curr_path and curr_time from state without passing too many params in train
    # possible to extend in the future
    curr_epoch = curr_state['epoch']
    curr_time = curr_state['time']
    #end 2
    
    #debug
    print('checkpoint step :',args.checkpoint_step)
    print('resnet :',args.context_path)
    print('epoch :', curr_epoch)
    #end
    
    # implementation of amp
    scaler = amp.GradScaler()

    for epoch in range(curr_epoch, args.num_epochs):

        # learning rate updating at each epoch
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        
        #start counting epoch
        tic = time.perf_counter()
        
        # model in training mode
        model.train()

        # set up the statistics on screen
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []

        # start inner loop for each batch
        for i, (data, label) in enumerate(dataloader_train):

            # resetting the optimizer's gradient
            optimizer.zero_grad()
            
            # passing batch on gpu
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # implementing autocast to speed up computations
            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            # backward pass
            scaler.scale(loss).backward()

            # save the loss as a numpy array
            loss = loss.data.cpu().numpy()

            # update the stats on screen
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            # update the scaler and the optimizer
            scaler.step(optimizer)
            scaler.update()

            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        
        # stop counting epoch
        toc = time.perf_counter()
        
        #update curr_time
        curr_time += toc-tic
        
        #debug time
        print('curr_time :', curr_time)
        #end
        
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        # save the checkpoint
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
                
            #new
            path = os.path.join(args.save_model_path, 'latest_crossentropy_loss_step2.pth')
            print('save model in %s ...' % path)
            state = {
                'epoch': epoch+1,
                'model_state': model.module.state_dict(),
                'optim_state': optimizer.state_dict(),
                'time': curr_time
            }
            torch.save(state,path)
            print('Done!')
            #end
        
        # save the best model
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_crossentropy_loss_step2.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    print('time avg :',curr_time/args.num_epochs)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=100, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    # save paths of data 
    train_path = [os.path.join(args.data, 'train'), os.path.join(args.data, 'val')]
    train_label_path = [os.path.join(args.data, 'train_labels'), os.path.join(args.data, 'val_labels')]
    test_path = os.path.join(args.data, 'test')
    test_label_path = os.path.join(args.data, 'test_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')

    # create dataset and dataloader for training
    dataset_train = CamVid(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                           loss=args.loss, mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # create dataset and dataloader for evaluation
    dataset_val = CamVid(test_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                         loss=args.loss, mode='test')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
        
    # initialize curr_state
    curr_state = {
        'epoch': 0,
        'time': 0 #seconds
    }

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        path = args.pretrained_model_path
        print('load model from %s ...' % path)
        state = torch.load(path)
        model.module.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optim_state'])
        curr_state['epoch'] = state['epoch']
        curr_state['time'] = state['time'] #seconds
        
        print('Done!')

    # train
    train(args, model, optimizer, curr_state, dataloader_train, dataloader_val)

    val(args, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '101',
        '--learning_rate', '2.5e-2',
        '--data', './data/CamVid',
        '--num_workers', '8',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '8',
        '--save_model_path', './checkpoints_101_sgd',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        
        # new parameters to try with checkpoints
        '--checkpoint_step', '5',
        '--validation_step', '5',
        '--loss', 'crossentropy',
        '--pretrained_model_path', './checkpoints_101_sgd/latest_crossentropy_loss_step2.pth',
    ]
    main(params)


