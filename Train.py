import torch
import socket
import time
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from PUIENet_MC import mynet
from torch.autograd import Variable
from data import get_training_set
from data import get_eval_set
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# Training settings
parser = argparse.ArgumentParser(description='PyTorch PUIE-Net')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--data_dir', type=str, default='dataset/new_UIEBD/train')
parser.add_argument('--label_train_dataset', type=str, default='label')
parser.add_argument('--data_train_dataset', type=str, default='image')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='10000', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--input_dir', type=str, default='dataset/new_UIEBD/test/image')
parser.add_argument('--output', default='results/UIEBD', help='Location to save checkpoint models')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--sample_out', type=str, default='sample')
parser.add_argument('--reference_out', type=str, default='mc')
parser.add_argument('--input_dir1', default='results/UIEBD/mc')
parser.add_argument('--reference_dir', default='dataset/new_UIEBD/test/label/O')
opt = parser.parse_args()
device = torch.device(opt.device)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, inputlab, target= Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.to(device)
            inputlab=inputlab.to(device)
            target = target.to(device)
            #targetlab = targetlab.to(device)
        t0 = time.time()        
        model.forward(input, inputlab, target, training=True)
        loss = model.elbo(target)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        t1 = time.time()

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec.".format(epoch, iteration,
                          #len(training_data_loader), loss.item(), optimizer.param_groups[0]['lr'], (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


# def test():
#     model.eval()
#     torch.set_grad_enabled(False)
#     for batch in testing_data_loader:
#         with torch.no_grad():
#             input, inputLAB, _, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), batch[3]
#         if cuda:
#             input = input.cuda(device)
#             inputLAB = inputLAB.cuda(device)
#
#         with torch.no_grad():
#
#             model.forward(input, inputLAB, input, training=False)
#             avg_pre = 0
#             for i in range(20):
#                 #t0 = time.time()
#                 prediction = model.sample(testing=True)
#                 #t1 = time.time()
#                 avg_pre = avg_pre + prediction / 20
#                 save_img_1(prediction.cpu().data, name[0], i, opt.sample_out)
#
#                 #print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
#             save_img_2(avg_pre.cpu().data, name[0], opt.reference_out)
#     im_path = opt.input_dir1
#     re_path = opt.reference_dir
#     avg_psnr = 0
#     avg_ssim = 0
#     n = 0
#     for filename in os.listdir(im_path):
#         print(im_path + '/' + filename)
#         n = n + 1
#         im1 = cv2.imread(im_path + '/' + filename)
#         im2 = cv2.imread(re_path + '/' + filename)
#
#         (h, w, c) = im2.shape
#         im1 = cv2.resize(im1, (w, h))  # reference size
#
#         score_psnr = psnr(im1, im2)
#         score_ssim = ssim(im1, im2, channel_axis=2, data_range=255)
#
#         avg_psnr += score_psnr
#         avg_ssim += score_ssim
#
#     avg_psnr = avg_psnr / n
#     avg_ssim = avg_ssim / n
#     print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
#     print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
def save_img_1(img, img_name, i, out):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(opt.output, out)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = img_name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '_' + str(i) + '.' + name_list[1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_img_2(img, img_name, out):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(opt.output, out)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = img_name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '.' + name_list[1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

train_set = get_training_set(opt.data_dir, opt.label_train_dataset, opt.data_train_dataset, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
test_set = get_eval_set(opt.input_dir, opt.input_dir)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
model = mynet(opt)

# print('---------- Networks architecture -------------')
# print_network(model)
# print('----------------------------------------------')

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)
                            
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    
    train(epoch)
#    test()
    scheduler.step()

    if (epoch+1) % opt.snapshots == 0:
        checkpoint(epoch)
