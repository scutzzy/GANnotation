import argparse
import yaml
from model.generator import Generator
from model.NLayerDiscriminator import NLayerDiscriminator
import torch
import torch.nn as nn
import torchvision.models as models

from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from data_loader import *
from PIL import Image
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
parser.add_argument('--gpu', type=int, default='0')
opts = parser.parse_args()

with open(opts.config) as stream:
    config = yaml.load(stream)
print(config)

device_ids = [0]
torch.cuda.set_device(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

writer = SummaryWriter()

# Load Data
dataset = ImageDataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
data_iter = iter(data_loader)

fix_input, fix_label, fix_target_1 = dataset.sample_fix()

netG = Generator(c_dim = 68)
netD = NLayerDiscriminator(input_nc = 3)

netG = nn.DataParallel(netG, device_ids=device_ids)
netG = netG.cuda()
netD = nn.DataParallel(netD, device_ids=device_ids)
netD = netD.cuda()

netG_Opt = torch.optim.Adam(netG.parameters(), lr = config['learning_rate'], betas=(config['beta1'], config['beta2']))
netD_Opt = torch.optim.Adam(netD.parameters(), lr = config['learning_rate'], betas=(config['beta1'], config['beta2']))

netG_scheduler = torch.optim.lr_scheduler.StepLR(netG_Opt, step_size=1, gamma=0.8)
netD_scheduler = torch.optim.lr_scheduler.StepLR(netD_Opt, step_size=1, gamma=0.8)




'''
Loss Function:

There are three different type of images:
1. Input Image (User Given)
2. Generated Image (Generated By Generator)
3. Target Image (Only For Training Stage, An Image With Its Own Landmark)

For Triple Consistence Loss, The Generated Image Is Divided Into Generated Image A And Generated Image B
'''
# Load VGG16 Model And Fix Its Paramater
vgg19 = models.vgg19(pretrained=True)
vgg19.eval()
vgg19 = torch.nn.DataParallel(vgg19, device_ids=device_ids).cuda()
vgg19.cuda()
for param in vgg19.parameters():
    param.requires_grad = False

# Load LightCNN
lightCNN = LightCNN_29Layers_v2(num_classes=80013)
lightCNN.eval()
lightCNN = torch.nn.DataParallel(lightCNN, device_ids=device_ids).cuda()
lightCNN.load_state_dict(torch.load("LightCNN_29Layers_v2_checkpoint.pth.tar")['state_dict'])
for param in lightCNN.parameters():
    param.requires_grad = False

MSELoss = nn.MSELoss(size_average=True).cuda()
L1Loss = nn.L1Loss(size_average=True).cuda()

def rgb_to_gray(i):
    img = (i[:,0,:,:] + i[:,1,:,:] + i[:,2,:,:]) / 3
    img = img.view(img.shape[0], 1, img.shape[1], img.shape[2])
    return img

def cal_pix_loss(generated_img, target_img):
    return MSELoss(generated_img, target_img) * config['dis']['loss_weight']['Pixel_Reconstruction']

def cal_total_variation_loss(generated_img):
    w_variance = torch.sum(generated_img[:,:,:,1:] - generated_img[:,:,:,:-1])
    h_variance = torch.sum(generated_img[:,:,1:,:] - generated_img[:,:,:-1,:])    
    return (w_variance + h_variance) / 2 * config['dis']['loss_weight']['Total_Variation']

def cal_consistency_loss(inverted_img, input_img):
    return L1Loss(inverted_img, input_img) * config['dis']['loss_weight']['Self_Consistency']

def cal_triple_consistency_loss(generated_img_a, generated_img_b):
    return L1Loss(generated_img_a, generated_img_b) * config['dis']['loss_weight']['Triple_Consistency']

def cal_identity_preserving_loss(generated_img, input_img):
    g_1, g_2 = lightCNN(rgb_to_gray(generated_img))
    i_1, i_2 = lightCNN(rgb_to_gray(input_img))
    loss_1 = L1Loss(g_1, i_1)
    loss_2 = L1Loss(g_2, i_2)
    return (loss_1 + loss_2) / 2 * config['dis']['loss_weight']['Identity_Preserving']


def cal_perceptual_loss(generated_img, target_img):
    # {relu1 2, relu2 2, relu3 3, relu4 3}
    hidden_output = []
    get_hidden_output = lambda m, i ,o : hidden_output.append(o)
    
    relu1_2 = vgg19.module.features._modules.get('3')
    relu2_2 = vgg19.module.features._modules.get('8')
    relu3_3 = vgg19.module.features._modules.get('15')
    relu4_3 = vgg19.module.features._modules.get('24')

    handle1 = relu1_2.register_forward_hook(get_hidden_output)
    handle2 = relu2_2.register_forward_hook(get_hidden_output)
    handle3 = relu3_3.register_forward_hook(get_hidden_output)
    handle4 = relu4_3.register_forward_hook(get_hidden_output)

    vgg19(generated_img)

    generated_img_hidden_output = hidden_output
    hidden_output = []

    vgg19(target_img)

    feature_reconstruction_loss = L1Loss(generated_img_hidden_output[0], hidden_output[0]) + \
        L1Loss(generated_img_hidden_output[1], hidden_output[1]) + \
        L1Loss(generated_img_hidden_output[2], hidden_output[2]) + \
        L1Loss(generated_img_hidden_output[3], hidden_output[3])

    feature_reconstruction_loss = feature_reconstruction_loss / 4

    a, b, c, d = generated_img_hidden_output[2].size()
    generated_img_features = generated_img_hidden_output[2].view(a * b, c * d)
    G_g = torch.mm(generated_img_features, generated_img_features.t())
    G_g = G_g.div(a * b * c * d)

    target_img_features = hidden_output[2].view(a * b, c * d)
    G_t = torch.mm(target_img_features, target_img_features.t())
    G_t = G_t.div(a * b * c * d)

    style_reconstruction_loss = torch.norm(G_g - G_t)

    handle1.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()
    return (style_reconstruction_loss + feature_reconstruction_loss) / 2 * config['dis']['loss_weight']['Perceptual']

def cal_adv_loss_dis(real, fake):
    loss = nn.ReLU()(1.0 - real).mean() + nn.ReLU()(1.0 + fake).mean()
    return loss

def cal_adv_loss_gen(fake):
    loss = -fake.mean()
    return loss

def cal_mask_loss(mask):
    return torch.mean(mask) * 6.5

  
# Start Training
for train_epoch in range(config['max_epoch']):
    for train_iter, data in enumerate(data_loader, 0):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        img_input, mask_input, img_real_1, mask_real_1, _, mask_real_2 = data
        img_input = img_input.cuda()
        mask_input = mask_input.cuda()
        img_real_1 = img_real_1.cuda()
        mask_real_1 = mask_real_1.cuda()
        mask_real_2 = mask_real_2.cuda()
        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        for _ in range(2):
            netD.zero_grad()
            # Compute loss with real images.
            d_real = netD(img_real_1)

            # Compute loss with fake images.
            img_generated, mask, _ = netG(torch.cat((img_input, mask_real_1), dim = 1))
            img_generated = img_generated.detach()
            d_fake = netD(img_generated)

            d_loss = cal_adv_loss_dis(d_real, d_fake)
            d_loss.backward()
            netD_Opt.step()

        # =================================================================================== #
        #                             3. Train the generator                                  #
        # =================================================================================== #
        netG.zero_grad()

        img_generated, mask, _ = netG(torch.cat((img_input, mask_real_1), dim = 1))

        img_generated_reverse,mask_generated_reverse, _ = netG(torch.cat((img_generated, mask_input), dim = 1))

        img_generated_triple_1,mask_generated_triple_1, _ = netG(torch.cat((img_input, mask_real_2), dim = 1))
        img_generated_triple_2,mask_generated_triple_2, _ = netG(torch.cat((img_generated, mask_real_2), dim = 1))
        
        g_mask_loss = cal_mask_loss(mask) + cal_mask_loss(mask_generated_reverse) + cal_mask_loss(mask_generated_triple_1) + cal_mask_loss(mask_generated_triple_2)
        g_adv_loss = cal_adv_loss_gen(img_generated)
        g_pix_loss = cal_pix_loss(img_generated, img_real_1)
        g_tv_loss = cal_total_variation_loss(img_generated)
        g_consistency_loss = cal_consistency_loss(img_generated_reverse, img_input)
        g_triple_consistency_loss = cal_triple_consistency_loss(img_generated_triple_1, img_generated_triple_2)
        g_identity_preserving_loss = cal_identity_preserving_loss(img_generated, img_input)
        g_perceptual_loss = cal_perceptual_loss(img_generated, img_input)

        g_loss = g_adv_loss + g_pix_loss + g_tv_loss + g_consistency_loss + g_triple_consistency_loss + g_identity_preserving_loss + g_perceptual_loss + g_mask_loss
        g_loss.backward()
        netG_Opt.step()

        if train_iter % 5 == 0:
            writer.add_scalar('pix_loss', g_pix_loss.data.cpu().numpy(), train_iter)
            writer.add_scalar('adv_loss', g_adv_loss.data.cpu().numpy(), train_iter)
            writer.add_scalar('tv_loss', g_tv_loss.data.cpu().numpy(), train_iter)
            writer.add_scalar('consistency_loss', g_consistency_loss.data.cpu().numpy(), train_iter)
            writer.add_scalar('triple_consistency_loss', g_triple_consistency_loss.data.cpu().numpy(), train_iter)
            writer.add_scalar('identity_preserving_loss', g_identity_preserving_loss.data.cpu().numpy(), train_iter)
            writer.add_scalar('perceptual_loss', g_perceptual_loss.data.cpu().numpy(), train_iter)
            _x = mask.data.cpu().numpy().mean()
            writer.add_scalar('mask_value', _x, train_iter)
            writer.add_scalar('mask_loss', g_mask_loss, train_iter)
            print('{}'.format(_x))
            
        if train_iter % 100 == 0:
            img_1,_,col_1 = netG(torch.cat((fix_input.view(1,3,128,128), fix_label.view(1,68,128,128)), dim = 1))
            writer.add_image('Image_Fix_1', 0.5*(img_1.data[0,:,:,:].cpu().numpy().swapaxes(0,1).swapaxes(1,2) + 1), train_iter,dataformats='HWC')
            writer.add_image('Image_Col_1', 0.5*(col_1.data[0,:,:,:].cpu().numpy().swapaxes(0,1).swapaxes(1,2) + 1), train_iter,dataformats='HWC')

        if train_iter % 50000 == 1:
            torch.save(netD.state_dict(), "./save/D_epoch-{}-{}".format(train_epoch,train_iter))
            torch.save(netG.state_dict(), "./save/G_epoch-{}-{}".format(train_epoch,train_iter))

    
    # Update Learning Rate At Each Epoch
    netD_scheduler.step()
    netG_scheduler.step()
    
    






