from typing import Any
from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

class GaussianErasing(object):
    
    def __init__(self,probability=0.5,sl=0.09,sh=0.7,rl=1,rh=3,thetal=-89,thetah=89,mean=[0.4914, 0.4822, 0.4465],density=0.5,scalar=10) -> None:
        '''     
        Implementation of GEA
        Param:
            probability: Probability of triggering GEA
            sl,sh: Legal erasing ratio range
            rl,rh: Variance ratio range
            thetal,thetah: Rotation range
            mean: [] value of replace pixel, default = [0.4914, 0.4822, 0.4465] 
            density: Density of Gaussian sampling
            scalar: Mapping scale of sampling
        '''
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = rh
        self.thetal = thetal
        self.thetah = thetah
        self.mean = mean
        self.density = density
        self.scalar = scalar
    
    def __call__(self, img) -> Any:
        if random.uniform(0,1) > self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            
            # generate parameters
            S_e = random.uniform(self.sl,self.sh) * area
            r_e = random.uniform(self.rl,self.rh)
            theta = random.uniform(self.thetal,self.thetah) * np.pi / 180   

            # generate gaussian kernel        
            w = np.sqrt(S_e / r_e) / 5         
            h = np.sqrt(S_e * r_e) / 5         
            sigma1 = 1              
            sigma2 = r_e * sigma1
            
            # generate legal erasing area
            phi = np.arctan(1 / r_e)
            hypotenuse = np.sqrt((5*w)**2+(5*h)**2)   
            if theta >= 0:
                W = hypotenuse * np.sin(theta+phi)
                H = hypotenuse * np.cos(theta-phi)
            else :
                _theta = -1 * theta
                W = hypotenuse * np.sin(_theta+phi)
                H = hypotenuse * np.cos(_theta-phi)
            W = int(W)
            H = int(H)
            
            # sampling with gaussian kernel 
            mask = generate_gaussian_mask(sigma1=sigma1,sigma2=sigma2,theta=theta,w=W,h=H,density=self.density,scalar=self.scalar)
                                 
            # pixel replacement
            if W // 2 < img.size()[2] and H // 2 < img.size()[1]:
                # generate expand image
                expand_img = torch.zeros(img.size()[0],img.size()[1]+H,img.size()[2]+W)
                expand_img[:,H//2:H//2+img.size()[1],W//2:W//2+img.size()[2]] = img[:,:,:]
                x1 = random.randint(0, expand_img.size()[1] - H)
                y1 = random.randint(0, expand_img.size()[2] - W)
                
                # erase pixel by mask
                if expand_img.size()[0] == 3:          
                    expand_img[:, x1:x1+H, y1:y1+W] = expand_img[:, x1:x1+H, y1:y1+W] * (1-mask)
                    add_mask = (torch.ones(mask.size())).repeat(3,1,1)
                    add_mask = add_mask * mask
                    add_mask[0,:,:] = add_mask[0,:,:] * self.mean[0]
                    add_mask[1,:,:] = add_mask[1,:,:] * self.mean[1]
                    add_mask[2,:,:] = add_mask[2,:,:] * self.mean[2]
                    expand_img[:, x1:x1+H, y1:y1+W] = expand_img[:, x1:x1+H, y1:y1+W] + add_mask
                else:
                    expand_img[0, x1:x1+H, y1:y1+W] = expand_img[0, x1:x1+H, y1:y1+W] * (1-mask)
                    add_mask = (torch.ones(mask.size())).repeat(1,1,1)
                    add_mask = add_mask * mask
                    add_mask[0,:,:] = add_mask[0,:,:] * self.mean[0]
                    expand_img[:, x1:x1+H, y1:y1+W] = expand_img[:, x1:x1+H, y1:y1+W] + add_mask
                img[:,:,:] = expand_img[:,H//2:H//2+img.size()[1],W//2:W//2+img.size()[2]]
                return img
        return img

def generate_gaussian_mask(sigma1,sigma2,theta,w,h,density=0.5,scalar=10):
    '''
    Implementation of Gaussian sampling
    Param:
        sigma1: variance of coordinate x
        sigma2: variance of coordinate y
        theta:  rotation angle
        w: width of legal erasing area
        h: height of legal erasing area
        density : sampling density
        scalar: scalar of coordinate mapping
    retrun:
        sampling mask
    '''
    w = int(w)
    h = int(h)
    
    if np.abs(theta) > 3.14:
        theta = theta * np.pi / 180
    
    # scalar matrix
    scalarMatrix=np.dot(np.matrix([[sigma1**2,0],[0,sigma2**2]]),np.identity(2))

    # rotation matrix
    rotationMatrix=np.matrix([[np.cos(theta),-1*np.sin(theta)],
                            [np.sin(theta),np.cos(theta)]])
    
    # covariance matrix
    covMatrix=np.dot(np.dot(rotationMatrix,scalarMatrix),rotationMatrix.transpose()) 
    
    # sample
    pts = np.random.multivariate_normal([0, 0], covMatrix, size=int(w*h*density))
    X = torch.Tensor(pts[:,0])      
    Y = torch.Tensor(pts[:,1])      
    locs = torch.stack((Y,X),dim=1)
    
    # mapping
    pts = (locs * scalar).int()            
    pts[:,0] = pts[:,0] + h //2     # h
    pts[:,1] = pts[:,1] + w //2     # w
    
    # filter illegal points
    select_mask = (pts[:,0]>=0)&(pts[:,0]<h)&(pts[:,1]>=0)&(pts[:,1]<w)
    pts = pts[select_mask]              
    pts = torch.unique(pts,dim=0)       
    
    # make the mask
    mask = torch.zeros(h,w)   
    lx = torch.LongTensor(pts[:,0].numpy()) 
    ly = torch.LongTensor(pts[:,1].numpy()) 
    replace_value = torch.ones_like(pts[:,0],dtype=mask.dtype)
    mask = mask.index_put((lx,ly),replace_value)
    
    return mask
        
if __name__ == '__main__':
    img = Image.open('../img/test.png')
    img.show()
    trans = transforms.Compose([
                transforms.ToTensor(),
                GaussianErasing(probability=1.0,thetal=0)]) 
    img1 = trans(img)
    img1 = transforms.ToPILImage()(img1)
    img1.show()

        