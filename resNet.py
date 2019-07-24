import  torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):

    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        #维度缩放
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        out = self.extra(x) +out

        return out


class ResNet18(nn.Module):

    def __init__(self,num_class):
        super(ResNet18,self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm2d(64)
        )
        # 跟着4个这个

        # 64 -----128
        self.blk1 = ResBlk(64,128,stride=3)
        #128 ______256
        self.blk2 = ResBlk(128,256,stride=3)
        #bian
        self.blk3 = ResBlk(256,512,stride=3)
        #跟！
        self.blk4 = ResBlk(512,512,stride=3)
        #注意此处的图片的长和宽都疯狂减少
        self.outlayer = nn.Linear(512*9,self.num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        #print("blk1",x.shape)
        x = self.blk2(x)
        #print("blk2", x.shape)
        x = self.blk3(x)
        #print("blk3", x.shape)
        x = self.blk4(x)
        #print("blk4", x.shape)


        #x = F.adaptive_avg_pool2d(x,[1,1])
        #print('after pool:',x.shape)
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x


def main():
    #blk = ResBlk(64,128,2)
    #tmp =torch.randn(2,64,224,224)
    #out = blk(tmp)

    #print(out.shape)
    model = ResNet18(2)

    tmp = torch.randn(2,3,224,224)
    out = model (tmp)
    print("adwufda",out.shape)
    p = sum(map(lambda p:p.numel(),model.parameters()))
    print("parameters ",p)


if __name__ == '__main__':
    main()
