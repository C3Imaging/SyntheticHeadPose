import time
import sys
import argparse
import datetime
import pathlib
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# Set seed
torch.manual_seed(0)

# Where to add a new import
from torch.optim.lr_scheduler import StepLR

# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
# from torchsummaryX import summary

import datasets
from HPEDA.FSANetDA import FSANet
from utils import AverageMeter


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Monocular Head Pose Estimation from Synthetic Data')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='epochs', help='Maximum number of training epochs.',
                        default=40, type=int)
    parser.add_argument('--bs', dest='batch_size', help='Batch size.',
                        default=8, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.0001, type=float)
    parser.add_argument("--validation_split", type=float, default=0.01,
                        help="validation split ratio")
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
                        default='/mnt/fastssd/Shubhajit_Stuff/HPECode/Data/BIWI/',
                        type=str)
    parser.add_argument('--filename_list', dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='/mnt/fastssd/Shubhajit_Stuff/HPECode/Data/Mixed/RawNPY.txt',
                        type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.',
                        default='BIWIRaw', type=str)

    # Pose_Synth_Raw | PoseSynthNPY
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_BIWI_NPY', type=str)

    args = parser.parse_args()
    return args


def main():

    # Parse Arguments
    args = parse_args()

    # get device GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    if args.dataset == 'PoseSynthNPY':
        pose_dataset = datasets.Pose_Synth_NPY(args.data_dir, args.filename_list)
    elif args.dataset == 'Pose_Synth_Raw':
        pose_dataset = datasets.Pose_Synth_Raw(args.data_dir, args.filename_list)
    elif args.dataset == 'PoseSynRealRaw':
        pose_dataset = datasets.Pose_Synth_Raw_RB(args.data_dir, args.filename_list)
    elif args.dataset == 'Pose_BIWI_NPY':
        pose_dataset = datasets.Pose_BIWI_NPY(args.data_dir, args.filename_list)
    elif args.dataset == 'Pose_Synth_NPYDA':
        pose_dataset = datasets.Pose_Synth_NPYDA(args.data_dir, args.filename_list)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    # hyper parameters & Model Params
    num_capsule = 3
    dim_capsule = 16
    routings = 5

    num_primcaps = 7 * 3
    m_dim = 5
    s_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model = FSANet(s_set).cuda()
    print('Model created.')

    # print(summary(model, torch.rand((1, 3, 64, 64)).cuda()))

    # get multiple GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # load model to GPU
    model.to(device)

    # transfer learning
    # modelPath = r'models/MySynthNPY_11-22-2020_21-41-04-n8857-e100-bs8-lr0.0001/weights.epoch89_model.pth'
    # model.load_state_dict(torch.load(modelPath))

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = args.batch_size

    # Load train loader
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=2)

    # Loss
    l1_criterion = nn.L1Loss()
    nll_criterion = nn.NLLLoss()  # domain adaptation

    now = datetime.datetime.now()  # current date and time
    runID = args.output_string + now.strftime("_%m-%d-%Y_%H-%M-%S") \
            + '-n' + str(len(train_loader)) \
            + '-e' + str(args.epochs) \
            + '-bs' + str(batch_size) \
            + '-lr' + str(args.lr)
    outputPath = './models/'
    runPath = outputPath + runID
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)

    # gamma = decaying factor (lr decayed on each step_size epoch with a rate of gamma
    # scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # print('###################################### Epoch :', str(epoch))

        # Switch to train mode
        model.train()
        end = time.time()

        # Decay Learning Rate
        # scheduler.step()

        for i, (images, cont_labels) in enumerate(train_loader):

            optimizer.zero_grad()

            images = Variable(images).cuda()
            label_angles = Variable(cont_labels[:, :3]).cuda(non_blocking=True)

            # Predict
            angles, _ = model(images, alpha=0.1)

            # Compute the loss
            l1_pose = l1_criterion(angles, label_angles)

            loss = l1_pose

            # Update step
            losses.update(loss.data.item(), images.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

            # Log progress
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'L1_Loss: {l1_loss:.4f}'
                      .format(epoch, i, N,
                              batch_time=batch_time,
                              loss=losses,
                              eta=eta,
                              l1_loss=l1_pose
                              ))

        # save Model intermediate
        path = runPath + '/weights.epoch{0}_model.pth'.format(epoch)
        torch.save(model.cpu().state_dict(), path)  # saving model
        model.cuda()

    # Start DA Training
    # for epoch in range(args.epochs):
    #     batch_time = AverageMeter()
    #     losses = AverageMeter()
    #     N = len(train_loader)
    #
    #     # print('###################################### Epoch :', str(epoch))
    #
    #     # Switch to train mode
    #     model.train()
    #
    #     end = time.time()
    #
    #
    #
    #     for i, (images, cont_labels, biwiImages) in enumerate(train_loader):
    #
    #         p = float(i + epoch * N) / args.epochs / N
    #         alpha = 2. / (1. + np.exp(-10 * p)) - 1
    #
    #         optimizer.zero_grad()
    #
    #         source_images = Variable(images).cuda()
    #         label_angles = Variable(cont_labels[:, :3]).cuda(non_blocking=True)
    #         source_domain_label = torch.zeros(batch_size)
    #         source_domain_label = source_domain_label.long().cuda()
    #
    #         target_images = Variable(biwiImages).cuda()
    #         target_domain_label = torch.ones(batch_size)
    #         target_domain_label = target_domain_label.long().cuda()
    #
    #         # Predict source domain
    #         angles, source_domain_output = model(source_images, alpha=alpha)
    #
    #         # Compute the loss in source domain
    #         l1_pose = l1_criterion(angles, label_angles)
    #         nll_source = nll_criterion(source_domain_output, source_domain_label)
    #
    #         # Predict target domain
    #         _, target_domain_output = model(target_images, alpha=alpha)
    #
    #         # Compute the loss in target domain
    #         nll_target = nll_criterion(target_domain_output, target_domain_label)
    #
    #         loss = 0.2*l1_pose + 1.5*nll_source + 1.5*nll_target
    #
    #
    #
    #         # Update step
    #         losses.update(loss.data.item(), images.size(0))
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #         eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))
    #
    #         # Log progress
    #         if i % 5 == 0:
    #             # Print to console
    #             print('Epoch: [{0}][{1}/{2}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
    #                   'ETA {eta}\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'L1_Loss: {l1_loss:.4f}'
    #                   .format(epoch, i, N,
    #                           batch_time=batch_time,
    #                           loss=losses,
    #                           eta=eta,
    #                           l1_loss=l1_pose
    #                           ))
    #
    #     # save Model intermediate
    #     path = runPath + '/weights.epoch{0}_model.pth'.format(epoch)
    #     torch.save(model.cpu().state_dict(), path)  # saving model
    #     model.cuda()


if __name__ == '__main__':
    main()

