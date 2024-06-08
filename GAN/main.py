import torch
import torch.nn.modules as nn
import torchvision as tv
from torch.autograd import Variable
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

dir = "./data"

noiseSize = 100  # 噪声维度
ngf = 64  # 生成器feature map数
ndf = 64  # 辨别器feature map数
batch_size = 256

d_every = 1  # 每一个batch训练一次discriminator
g_every = 5  # 每五个batch训练一次generator


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.main = nn.Sequential(
            # 输入为100维的噪声向量 计算公式：(in-1)*stride-2*padding+kernel
            nn.ConvTranspose2d(noiseSize, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输出：ngf*8 * 4 * 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输出：ngf*4 * 8 * 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输出：ngf*2 * 16 * 16

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 输出：ngf * 16 * 16

            nn.ConvTranspose2d(ngf, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
            # 3 * 96 * 96
        )

    def forward(self, input):
        return self.main(input)


class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.main = nn.Sequential(
            # 输出形状：(in+2*padding-kernel)/stride+1
            # 输入形状：3*96*96
            nn.Conv2d(3, ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：ndf*32*32

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：ndf*2 * 16 * 16

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：ndf*4 * 8 * 8

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：ndf*8 * 4 * 4

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 输出一个概率
        )

    def forward(self, input):
        return self.main(input).view(-1)  # 使用 view(-1) 将张量转换为一维行向量

def train():
    for i,(image,_) in tqdm.tqdm(enumerate(dataloader)):
        real_image = Variable(image)
        real_image = real_image.cuda()

        if (i + 1) % d_every == 0:
            opt_d.zero_grad()
            output = discriminator(real_image)  # 尽可能把真图片判为True
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()

            noises.data.copy_(torch.randn(batch_size, noiseSize, 1, 1))
            fake_img = generator(noises).detach()  # 根据噪声生成假图
            fake_output = discriminator(fake_img)  # 尽可能把假图片判为False
            error_d_fake = criterion(fake_output, fake_labels)
            error_d_fake.backward()
            opt_d.step()

        if (i + 1) % g_every == 0:
            opt_g.zero_grad()
            noises.data.copy_(torch.randn(batch_size, noiseSize, 1, 1))
            fake_img = generator(noises)  # 这里没有detach
            fake_output = discriminator(fake_img)  # 尽可能让Discriminator把假图片判为True
            error_g = criterion(fake_output, true_labels)
            error_g.backward()
            opt_g.step()

# 图像显示
def show(num):
    fix_fake_imags = generator(fix_noises)
    fix_fake_imags = fix_fake_imags.data.cpu()[:64] * 0.5 + 0.5

    fig = plt.figure(1)

    i = 1
    for image in fix_fake_imags:
        # 在一个 8x8 的子图网格中添加一个子图，并且子图的位置由变量 i 决定。通常情况下
        ax = fig.add_subplot(8, 8, i)
        # 去除了每个子图的坐标轴
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0))
        i += 1
    # 调整子图之间的间距 left子图左边界到整个图像左边界的距离，wspace子图之间的水平间距
    plt.subplots_adjust(left=None,  right=None,  bottom=None,  top=None,
                        wspace=0.05,  hspace=0.05)
    plt.suptitle('第%d迭代结果' % num, y=0.91, fontsize=15)
    plt.show()


if __name__ == '__main__':
    transform = tv.transforms.Compose([
        tv.transforms.Resize(96),
        tv.transforms.CenterCrop(96),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(root='data/',transform=transform)

    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    print("数据加载完毕")

    generator = NetG()
    discriminator = NetD()

    # 优化器
    opt_g = torch.optim.Adam(generator.parameters(),lr=2e-4,betas=(0.5,0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # 创建一些张量，让变量可以在GPU上使用
    true_labels=Variable(torch.ones(batch_size))
    fake_labels = Variable(torch.zeros(batch_size))
    fix_noises = Variable(torch.randn(batch_size,noiseSize,1,1))
    noises = Variable(torch.randn(batch_size,noiseSize,1,1))

    if torch.cuda.is_available() == True:
        print('cuda is available')
        generator.cuda()
        discriminator.cuda()
        criterion.cuda()
        true_labels,fake_labels = true_labels.cuda(),fake_labels.cuda()
        fix_noises,noises = fix_noises.cuda(),noises.cuda()

    plot_epoch = [1,5,9]

    for i in range(10):        # 最大迭代次数
        train()
        print('迭代次数：{}'.format(i))
        if i in plot_epoch:
            show(i)

