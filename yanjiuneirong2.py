from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
from PIL import ImageFile
from torchsummary import summary
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
# from IPython.display import HTML

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+2, ngf * 8, 4, 1, 0, bias=False),
            nn.SELU(True),
            # state size. (ngf*16) x 4 x 4
            # nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            # nn.SELU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.SELU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.SELU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.SELU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            # nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d((2, 2)),

            # nn.Linear(4 * 4 * 1024, 2),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(1024, 2),
            # nn.Sigmoid()
            # state size. (ndf*16) x 4 x 4
            # nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            # nn.Sigmoid()
            # state size. 1
        )
        self.input_sig = nn.Sequential(
            # nn.AvgPool2d(),
            nn.Linear(4 * 4 * 512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )

    def forward(self, input):
        res = self.main(input)
        res = res.view(res.size(0), -1)
        res = self.input_sig(res)
        return res
        # return self.main(input)

def showimg(images, count):
    images = images.to('cpu')
    images = images.detach().numpy()
    images = images[[6, 12, 18, 24, 30, 36, 42]]
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    grid_length = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(4, 4))
    width = images.shape[2]
    print('width是：',width)
    gs = gridspec.GridSpec(grid_length, grid_length, wspace=0, hspace=0)
    for i, img in enumerate(images):
        print(len(img),'aaa',len(img[0]))
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(3,width, width), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
    #  plt.tight_layout()
    plt.savefig(r'/mnt/data/Liuzhuoyue/biyesheji/CGAN/images/%d.png' % count, bbox_inches='tight')

if __name__ == '__main__':
    #dataroot = "data/img_align_celeba"
    dataroot = "/mnt/data/Liuzhuoyue/8bit/yjnr2/train"
    # Number of workers for dataloader
    workers = 8
    batch_size = 64
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 128
    # Size of feature maps in discriminator
    ndf = 64
    num_epochs = 200
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    ngpu = 1
    gepoch = 1
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomVerticalFlip(p=0.5),
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers, drop_last=True)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    #device = torch.device('cuda',0)

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # Create the generator
    G = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(G, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    G.apply(weights_init)

    # Print the model
    print(G)

    D = Discriminator(ngpu).to(device)
    summary(D, (3, 64, 64))

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(D, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    D.apply(weights_init)

    # Print the model
    print(D)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    BCE_stable = torch.nn.BCEWithLogitsLoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    count = 0
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (img, label) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            labels_onehot = np.zeros((batch_size, 2))
            # print(batch_size)
            # print(label.numpy())
            labels_onehot[np.arange(batch_size), label.numpy()] = 1

            #    img=img.view(num_img,-1)
            #    img=np.concatenate((img.numpy(),labels_onehot))
            #    img=torch.from_numpy(img)
            img = img.to(device)
            real_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()  # 真实label为1
            # print(real_label)
            fake_label = Variable(torch.zeros(batch_size, 2)).cuda()  # 假的label为0
            # print('这是假的lable', fake_label)

            # compute loss of real_img
            real_out = D(img)  # 真实图片送入判别器D输出0~1
            d_loss_real = criterion(real_out, real_label)  # 得到loss
            real_scores = real_out  # 真实图片放入判别器输出越接近1越好

            # compute loss of fake_img
            z = Variable(torch.randn(batch_size, 102, 1, 1, device=device))  # 随机生成向量
            # print('这是初步的z的大小', z.size())
            # print('这是z的大小：', z.size(), '这是img的大小：', batch_size)
            # print('这是z的数据：', z)
            fake_img = G(z)  # 将向量放入生成网络G生成一张图片
            fake_out = D(fake_img)  # 判别器判断假的图片
            d_loss_fake = criterion(fake_out, fake_label)  # 假的图片的loss
            fake_scores = fake_out  # 假的图片放入判别器输出越接近0越好

            # D bp and optimize
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()  # 判别器D的梯度归零
            d_loss.backward()  # 反向传播
            d_optimizer.step()  # 更新判别器D参数

            # 生成器G的训练compute loss of fake_img
            for j in range(gepoch):
                z = torch.randn(batch_size, 100, 1, 1)  # 随机生成向量
                labels_onehot_G = np.zeros((batch_size, 2 , 1 , 1))
                labels_onehot_G[np.arange(batch_size), label.numpy()] = 1
                # print(labels_onehot_G)
                z = np.concatenate((z.numpy(), labels_onehot_G), axis=1)
                z = Variable(torch.from_numpy(z).float()).cuda()
                # print('这是后来的z的大小', z.size())
                fake_img = G(z)  # 将向量放入生成网络G生成一张图片
                output = D(fake_img)  # 经过判别器得到结果
                g_loss = criterion(output, real_label)  # 得到假的图片与真实标签的loss
                # bp and optimize
                g_optimizer.zero_grad()  # 生成器G的梯度归零
                g_loss.backward()  # 反向传播
                g_optimizer.step()  # 更新生成器G参数
                temp = real_label
        if (i % 10 == 0) and (i != 0):
            torch.save(G.state_dict(), r'/mnt/data/Liuzhuoyue/biyesheji/CGAN/Generator_cuda_%d.pkl' % i)
            torch.save(D.state_dict(), r'/mnt/data/Liuzhuoyue/biyesheji/CGAN/Discriminator_cuda_%d.pkl' % i)
            # save_model(G, r'/mnt/data/Liuzhuoyue/biyesheji/CGAN/Generator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型
            # save_model(D, r'/mnt/data/Liuzhuoyue/biyesheji/CGAN/Discriminator_cpu_%d.pkl' % i)  # 保存为CPU中可以打开的模型
        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
              'D real: {:.6f}, D fake: {:.6f}'.format(
            num_epochs, epoch, d_loss.item(), g_loss.item(),
            real_scores.data.mean(), fake_scores.data.mean()))
        temp = temp.to('cpu')
        _, x = torch.max(temp, 1)
        x = x.numpy()
        fake = fake_img.detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=0, normalize=True))
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig('/mnt/data/Liuzhuoyue/8bit/res/2/' + str(epoch) + "per_air_a_river.jpg")
        # print(x)
        # print(x[[6, 12, 18, 24, 30, 36, 42]])
        # print(fake_img.size())
        # showimg(fake_img, count)
        # plt.show()
        count += 1


