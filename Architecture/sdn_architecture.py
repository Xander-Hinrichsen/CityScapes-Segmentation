import torch
import torch.nn as nn
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DenseEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            DenseNet = torchvision.models.densenet161(weights="IMAGENET1K_V1").features[:9]
        else:
            DenseNet = torchvision.models.densenet161().features[:9]
        self.biggest = DenseNet[:5]
        self.medium = DenseNet[5:7]
        self.small = DenseNet[7:9]
        self.compression = nn.Conv2d(2112, 1024, kernel_size=(3,3), padding=1)
    def forward(self, xb):
        big = self.biggest(xb)
        medium = self.medium(big)
        small = self.small(medium)
        small = self.compression(small)
        return small, medium, big

##test - the output of our densenet should be of shapes ([1, ?, 32, 64], [1, 2112,64, 128])
## and [1, 2112,64, 128]
# DenseEncoderTest = DenseEncoder(DenseNet)
# first, second, third = DenseEncoderTest(test)
# first.shape, second.shape, third.shape

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, isnormal=False):
        super().__init__()
        if not isnormal:
            self.network = nn.Sequential(
                nn.BatchNorm2d(in_size),
                nn.ReLU(),
                nn.Conv2d(in_size, out_size, kernel_size=(3,3),padding=1)
                #nn.Dropout2d(p=0.2)
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=(3,3),padding=1),
                nn.BatchNorm2d(in_size),
                nn.ReLU()
                #nn.Dropout2d(p=0.2)
            )
    def forward(self, xb):
        return self.network(xb)

class DownBlockMidRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = ConvBlock(576+768, 48)
        self.conv2 = ConvBlock((576+768)+48, 48)
        self.compression = nn.Conv2d(576+768 + 2*(48),768, kernel_size=(3,3), padding=1)
    def forward(self, xb, inter):
        xb = self.pool(xb)
        in_conv1 = torch.cat((xb, inter), dim=1)
        out_conv1 = self.conv1(in_conv1)
        in_conv2 = torch.cat((in_conv1, out_conv1),dim=1)
        out_conv2 = self.conv2(in_conv2)
        in_compression = torch.cat((in_conv2,out_conv2),dim=1)
        del(xb)
        del(in_conv1);del(in_conv2)
        del(out_conv1);del(out_conv2)
        torch.cuda.empty_cache()
        return self.compression(in_compression)

class DownBlockLowRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = ConvBlock(1024+768,48)
        self.conv2 = ConvBlock(1840,48)
        self.conv3 = ConvBlock(1888,48)
        self.conv4 = ConvBlock(1936,48)
        self.compression = nn.Conv2d(1024+768 + (4*48),1024, kernel_size=(3,3), padding=1)
    def forward(self, xb, inter):
        xb = self.pool(xb)
        in_conv1 = torch.cat((xb,inter),dim=1)
        out_conv1 = self.conv1(in_conv1)
        in_conv2 = torch.cat((in_conv1, out_conv1), dim=1)
        out_conv2 = self.conv2(in_conv2)
        in_conv3 = torch.cat((in_conv2, out_conv2), dim=1)
        out_conv3 = self.conv3(in_conv3)
        in_conv4 = torch.cat((in_conv3, out_conv3), dim=1)
        out_conv4 = self.conv4(in_conv4)
        in_compression = torch.cat((in_conv4, out_conv4), dim=1)
        del(xb)
        del(in_conv1);del(in_conv2);del(in_conv3);del(in_conv4);
        del(out_conv1);del(out_conv2);del(out_conv3);del(out_conv4);
        torch.cuda.empty_cache()
        return self.compression(in_compression)

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, no_deconv=False):
        super().__init__()
        if no_deconv:
            self.Upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1))
        else:
            self.Upsampler = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=(4,4), stride=2, padding=1)
    def forward(self, xb):
        return self.Upsampler(xb)

class UpBlockLowRes(nn.Module):
    def __init__(self,no_deconv=False):
        super().__init__()
        self.upsampler = Upsampler(1024,1024,no_deconv=no_deconv)
        self.conv1 = ConvBlock(1024+768, 48)
        self.conv2 = ConvBlock(1024+768+48, 48)
        self.compression = nn.Conv2d(1024+768+2*(48),768, kernel_size=(3,3), padding=1)
    def forward(self, xb, densenet_inter):
        xb = self.upsampler(xb)
        in_conv1 = torch.cat((xb,densenet_inter), dim=1)
        out_conv1 = self.conv1(in_conv1)
        in_conv2 = torch.cat((in_conv1, out_conv1),dim=1)
        out_conv2 = self.conv2(in_conv2)
        in_compression = torch.cat((in_conv2,out_conv2),dim=1)
        del(xb)
        del(in_conv1);del(in_conv2)
        del(out_conv1);del(out_conv2)
        torch.cuda.empty_cache()
        return self.compression(in_compression)

# asdf = UpBlockLowRes(no_deconv=True)
# inter = torch.ones((1,768,64,128),dtype=torch.float)
# xb = torch.ones((1,1024,32,64),dtype=torch.float)
# asdf(xb,inter).shape

class UpBlockMidRes(nn.Module):
    def __init__(self,no_deconv=False, to_compress=True):
        super().__init__()
        self.upsampler = Upsampler(768,768,no_deconv=no_deconv)
        self.conv1 = ConvBlock(768+384, 48)
        self.conv2 = ConvBlock(768+384+48, 48)
        self.compression = nn.Conv2d(768+384+2*(48),576, kernel_size=(3,3), padding=1)
        self.to_compress = to_compress
    def forward(self, xb, densenet_inter):
        xb = self.upsampler(xb)
        in_conv1 = torch.cat((xb,densenet_inter), dim=1)
        out_conv1 = self.conv1(in_conv1)
        in_conv2 = torch.cat((in_conv1, out_conv1),dim=1)
        out_conv2 = self.conv2(in_conv2)
        
        if not self.to_compress:
            del(xb)
            del(in_conv1)
            del(out_conv1)
            torch.cuda.empty_cache()
            return torch.cat((in_conv2,out_conv2),dim=1)
        
        in_compression = torch.cat((in_conv2,out_conv2),dim=1)
        del(xb)
        del(in_conv1);del(in_conv2)
        del(out_conv1);del(out_conv2)
        torch.cuda.empty_cache()
        return self.compression(in_compression)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownBlockMidRes()
        self.down2 = DownBlockLowRes()
    def forward(self, xb, inter1, inter2):
        out1 = self.down1(xb, inter1)
        out1 = self.down2(out1, inter2)
        return out1

class Decoder(nn.Module):
    def __init__(self, no_deconv=False, to_compress=True):
        super().__init__()
        self.up1 = UpBlockLowRes(no_deconv=no_deconv)
        self.up2 = UpBlockMidRes(no_deconv=no_deconv,to_compress=to_compress)
    def forward(self, xb, densenet_inter1, densenet_inter2):
        out1 = self.up1(xb, densenet_inter1)
        out2 = self.up2(out1, densenet_inter2)
        return out1, out2

class SuperVision(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3, scale_factor, num_classes, retfinal=False):
        super().__init__()
        self.classifier1 = nn.Conv2d(in_channels1, num_classes, kernel_size=(3,3), padding=1)
        self.classifier2 = nn.Conv2d(in_channels2, num_classes, kernel_size=(3,3), padding=1)
        self.classifier3 = nn.Conv2d(in_channels3, num_classes, kernel_size=(3,3), padding=1)
        
        self.upscaler1 = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.upscaler2 = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.upscaler3 = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.retfinal = retfinal
    def forward(self, first, second, third): 
        if not self.retfinal:
            first = self.classifier1(first)
            second = self.classifier2(second) + first
            third = self.classifier3(third) + second
            first = self.upscaler1(first)
            second = self.upscaler2(second)
            third = self.upscaler3(third)
            return first, second, third
        first = self.classifier1(first)
        second = self.classifier2(second) + first
        final = self.classifier3(third)
        third = final + second
        first = self.upscaler1(first)
        second = self.upscaler2(second)
        third = self.upscaler3(third)
        final = self.upscaler3(final)
        return final, (first, second, third)

class SDN(nn.Module):
    def __init__(self,no_deconv=False,num_classes=20, pretrained=False):
        super().__init__()
        self.densenet = DenseEncoder(pretrained=pretrained)
        self.decoder1 = Decoder(no_deconv=no_deconv)
        self.encoder2 = Encoder()
        self.decoder2 = Decoder(no_deconv=no_deconv)
        self.encoder3 = Encoder()
        self.decoder3 = Decoder(no_deconv=no_deconv,to_compress=False)
        
        self.supervise_lowres = SuperVision(1024,1024,1024,16,num_classes)
        self.supervise_midres = SuperVision(768,768,768,8, num_classes)
        self.supervise_highres = SuperVision(576,576,1248,4, num_classes,retfinal=True)
    def forward(self, xb):
        dense_out, dense_mid_res, dense_high_res = self.densenet(xb)
        decoder1_low, decoder1_high = self.decoder1(dense_out,dense_mid_res,dense_high_res)
        encoder2 = self.encoder2(decoder1_high, decoder1_low, dense_out)
        decoder2_low, decoder2_high = self.decoder2(encoder2, dense_mid_res,dense_high_res)
        encoder3 = self.encoder3(decoder2_high, decoder2_low, encoder2)
        decoder3_low, decoder3_high = self.decoder3(encoder3, dense_mid_res,dense_high_res)
        del(dense_mid_res); del(dense_high_res)
        torch.cuda.empty_cache()
        supervised_lowres = self.supervise_lowres(dense_out, encoder2, encoder3)
        supervised_midres = self.supervise_midres(decoder1_low, decoder2_low, decoder3_low)
        final, supervised_highres = self.supervise_highres(decoder1_high, decoder2_high, decoder3_high)
        del(dense_out); del(encoder2);del(encoder3)
        del(decoder1_low);del(decoder2_low);del(decoder3_low)
        del(decoder1_high);del(decoder2_high);del(decoder3_high)
        torch.cuda.empty_cache()
        return final, supervised_lowres, supervised_midres, supervised_highres

class Supervised_CE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crossentropy = nn.CrossEntropyLoss()
    def forward(self,final, supervise1, supervise2, supervise3, ground_truth):
        #loss_lowres1 = self.crossentropy(supervise1[0], ground_truth)
        #loss_lowres2 = self.crossentropy(supervise1[1], ground_truth)
        #loss_lowres3 = self.crossentropy(supervise1[2], ground_truth)
        #final = self.crossentropy(final, ground_truth)
        loss_midres1 = self.crossentropy(supervise2[0], ground_truth)
        loss_midres2 = self.crossentropy(supervise2[1], ground_truth)
        loss_midres3 = self.crossentropy(supervise2[2], ground_truth)
        loss_highres1 = self.crossentropy(supervise3[0], ground_truth)
        loss_highres2 = self.crossentropy(supervise3[1], ground_truth)
        loss_highres3 = self.crossentropy(supervise3[2], ground_truth)
        total_loss = (loss_midres1 + loss_midres2 + loss_midres3 +
                loss_highres1 + loss_highres2 + loss_highres3)

        del(supervise1); del(supervise2); del(supervise3); del(ground_truth)
        torch.cuda.empty_cache()
        return total_loss
