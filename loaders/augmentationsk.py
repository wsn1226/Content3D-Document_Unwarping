import os
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import tqdm
import random


def tight_crop(im, fm):
    # different tight crop
    msk=((fm[:,:,0]==0)&(fm[:,:,1]==0)&(fm[:,:,2]==0)).astype(np.uint8)
    msk=1-msk
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]
    fm = fm[miny : maxy + 1, minx : maxx + 1, :]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    fm = np.pad(fm, ((s, s), (s, s), (0, 0)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    fm = fm[cy1 : -cy2, cx1 : -cx2, :]

    return im, fm

def tight_crop_boundary(im, dm):
    # different tight crop
    msk=(dm!=0).astype(np.uint8)
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]
    dm = dm[miny : maxy + 1, minx : maxx + 1]

    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    dm = np.pad(dm, ((s, s), (s, s)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    dm = dm[cy1 : -cy2, cx1 : -cx2]
    return im, dm

def tight_crop_boundary_joint_masked_depth(im, boundary, dmap):
    # different tight crop
    msk=(boundary!=0).astype(np.uint8)
    size=msk.shape
    dmap=dmap*msk
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    maxd=np.max(dmap[msk.nonzero()])
    mind=np.min(dmap[msk.nonzero()])
    dmap[msk.nonzero()]=(dmap[msk.nonzero()]-mind)/(maxd-mind)+1

    im = im[miny : maxy + 1, minx : maxx + 1, :]
    boundary = boundary[miny : maxy + 1, minx : maxx + 1]
    dmap = dmap[miny : maxy + 1, minx : maxx + 1]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    boundary = np.pad(boundary, ((s, s), (s, s)), 'constant')
    dmap = np.pad(dmap, ((s, s), (s, s)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    boundary = boundary[cy1 : -cy2, cx1 : -cx2]
    dmap= dmap [cy1 : -cy2, cx1 : -cx2]
    t=miny-s+cy1
    b=size[0]-maxy-s+cy2
    l=minx-s+cx1
    r=size[1]-maxx-s+cx2

    return im, boundary,dmap,t,b,l,r


def data_aug_boundary(im, fm, bg):
    im=im/255.0
    bg=bg/255.0
    im, fm=tight_crop_boundary(im, fm)
    # change background img
    # msk = fm[:, :, 0] > 0
    if fm.shape[-1]==3:
        msk=msk=1-((fm[:,:,0]==0)&(fm[:,:,1]==0)&(fm[:,:,2]==0)).astype(np.uint8)
    else:
        # For depth
        msk=(fm!=0).astype(np.uint8)
    msk = np.expand_dims(msk, axis=2)
    # replace bg
    fh, fw, _ = im.shape
    chance=random.random()
    if chance > 0.3:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    elif chance < 0.3 and chance> 0.2:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    else:
        bg=np.zeros((fh, fw, 3))
        msk=np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    return im, fm

def data_aug_joint_masked_depth(im, boundary, dmap, bg):
    im=im/255.0
    bg=bg/255.0
    im, boundary,dmap ,t,b,l,r=tight_crop_boundary_joint_masked_depth(im, boundary,dmap)
    # change background img
    # msk = fm[:, :, 0] > 0
    msk=(boundary!=0).astype(np.uint8)
    msk = np.expand_dims(msk, axis=2)
    # replace bg
    fh, fw, _ = im.shape
    chance=random.random()
    if chance > 0.5:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    elif chance < 0.5 and chance> 0.35:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    else:
        bg=np.zeros((fh, fw, 3))
        msk=np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    # im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    return im, boundary,dmap, t,b,l,r


def tight_crop_depth_joint_masked_depth(im, dmap):
    # different tight crop
    msk=(dmap<10000).astype(np.uint8)
    size=msk.shape
    dmap=dmap*msk
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    maxd=np.max(dmap[msk.nonzero()])
    mind=np.min(dmap[msk.nonzero()])
    dmap[msk.nonzero()]=(dmap[msk.nonzero()]-mind)/(maxd-mind)+1

    im = im[miny : maxy + 1, minx : maxx + 1, :]
    dmap = dmap[miny : maxy + 1, minx : maxx + 1]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    dmap = np.pad(dmap, ((s, s), (s, s)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    dmap= dmap [cy1 : -cy2, cx1 : -cx2]
    t=miny-s+cy1
    b=size[0]-maxy-s+cy2
    l=minx-s+cx1
    r=size[1]-maxx-s+cx2
    return im,dmap,t,b,l,r

def nm_tight_crop_joint(im, norm):
    # different tight crop
    msk=(1-((norm[:,:,0]==0)&(norm[:,:,1]==0)&(norm[:,:,2]==0))).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    im = im[miny : maxy + 1, minx : maxx + 1, :]
    norm = norm[miny : maxy + 1, minx : maxx + 1, :]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    norm = np.pad(norm, ((s, s), (s, s), (0, 0)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    norm= norm[cy1 : -cy2, cx1 : -cx2, :]

    t=miny-s+cy1
    b=size[0]-maxy-s+cy2
    l=minx-s+cx1
    r=size[1]-maxx-s+cx2
    return im,norm,t,b,l,r

def data_aug_full_joint(im, dmap, bg):
    im=im/255.0
    bg=bg/255.0
    im,dmap,t,b,l,r=tight_crop_depth_joint_masked_depth(im,dmap)
    # change background img
    # msk = fm[:, :, 0] > 0
    msk=(dmap>=1).astype(np.uint8)
    msk = np.expand_dims(msk, axis=2)
    # replace bg
    fh, fw, _ = im.shape
    chance=random.random()
    if chance > 0.125:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    else:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    #else:
        #bg=np.zeros((fh, fw, 3))
        #msk=np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    return im, dmap, t,b,l,r

def nm_data_aug_full_joint(im, norm, bg):
    im=im/255.0
    bg=bg/255.0
    im,norm,t,b,l,r=nm_tight_crop_joint(im,norm)
    # change background img
    # msk = fm[:, :, 0] > 0
    msk=(1-((norm[:,:,0]==0)&(norm[:,:,1]==0)&(norm[:,:,2]==0))).astype(np.uint8)
    msk = np.expand_dims(msk, axis=2)
    # replace bg
    fh, fw, _ = im.shape
    chance=random.random()
    if chance > 0.125:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    else:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    #else:
        #bg=np.zeros((fh, fw, 3))
        #msk=np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    return im, norm, t,b,l,r




# def main():
#     tex_id=random.randint(1,5640)
#     with open(os.path.join(root[:-7],'augtexnames.txt'),'r') as f:
#         for i in range(tex_id):
#             txpth=f.readline().strip()

#     for im_name in filenames:
        
#         im_path = os.path.join(root,'img',im_name+'.png')
#         img=cv2.imread(im_path).astype(np.uint8)
        
#         lbl_path = os.path.join(root, 'wc',im_name+'.exr')
#         lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

#         tex=cv2.imread(os.path.join(root[:-7],txpth)).astype(np.uint8)
#         bg=cv2.resize(tex,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LANCZOS4)

#         img,lbl=data_aug(img,lbl,bg)

# if __name__ == '__main__':
#     main()

def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    f = random.uniform(-hue, hue)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + f * 360, 0., 360.)
    #= random.uniform(-saturation, saturation)
    #hsv[:,:,1] = np.clip(hsv[:,:,1] + f, 0., 1.)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    im = np.clip(im, 0., 1.)
    return im

def change_intensity(img):
    chance=random.uniform(0,1)
    # print(chance)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if chance>0.3:
        inc=random.randint(15,50)
        # print(inc)
        #increase
        v = nimg[:, :, 2]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 2] = v

    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg


def change_hue_sat(img):
    chance=random.uniform(0,1)
    # print(chance)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if chance>0.3:
        inc=random.randint(5,15)
        # print(inc)
        #increase
        v = nimg[:, :, 0]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 0] = v

    if chance>0.3:
        inc=random.randint(5,15)
        # print(inc)
        #increase
        v = nimg[:, :, 1]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 1] = v
    
    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg

def tight_crop_depth_joint_masked_depth_withrecon(im,recon, dmap):
    # different tight crop
    msk=(dmap<10000).astype(np.uint8)
    size=msk.shape
    dmap=dmap*msk
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    maxd=np.max(dmap[msk.nonzero()])
    mind=np.min(dmap[msk.nonzero()])
    dmap[msk.nonzero()]=(dmap[msk.nonzero()]-mind)/(maxd-mind)+1

    im = im[miny : maxy + 1, minx : maxx + 1, :]
    dmap = dmap[miny : maxy + 1, minx : maxx + 1]
    recon=recon[miny : maxy + 1, minx : maxx + 1, :]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
    dmap = np.pad(dmap, ((s, s), (s, s)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    recon= recon [cy1 : -cy2, cx1 : -cx2, :]
    dmap= dmap [cy1 : -cy2, cx1 : -cx2]
    t=miny-s+cy1
    b=size[0]-maxy-s+cy2
    l=minx-s+cx1
    r=size[1]-maxx-s+cx2
    return im,recon,dmap,t,b,l,r


def data_aug_full_joint_withrecon(im,recon,dmap, bg):
    im=im/255.0
    recon=recon/255.0
    bg=bg/255.0
    im,recon,dmap,t,b,l,r=tight_crop_depth_joint_masked_depth_withrecon(im,recon,dmap)
    # change background img
    # msk = fm[:, :, 0] > 0
    msk=(dmap>=1).astype(np.uint8)
    msk = np.expand_dims(msk, axis=2)
    # replace bg
    fh, fw, _ = im.shape
    chance=random.random()
    if chance > 0.3:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    elif chance < 0.3 and chance> 0.2:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    else:
        bg=np.zeros((fh, fw, 3))
        msk=np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    recon = color_jitter(recon, 0.2, 0.2, 0.6, 0.6)
    if chance>0.666:
        recon=recon[:,:,[1,2,0]] #BRG
    elif chance>0.333 and chance <=0.666:
        recon=recon[:,:,[2,1,0]] #BGR
    return im, recon, dmap, t,b,l,r

def data_aug_full_joint_withrecon_normald(im,recon,dmap, bg):
    im=im/255.0
    recon=recon/255.0
    bg=bg/255.0
    im,recon,dmap,t,b,l,r=tight_crop_depth_joint_masked_depth_withrecon_normald(im,recon,dmap)
    # change background img
    # msk = fm[:, :, 0] > 0
    msk=(dmap!=0).astype(np.uint8)
    msk = np.expand_dims(msk, axis=2)
    # replace bg
    fh, fw, _ = im.shape
    chance=random.random()
    if chance > 0.3:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    elif chance < 0.3 and chance> 0.2:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    else:
        bg=np.zeros((fh, fw, 3))
        msk=np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    if chance>0.666:
        recon=recon[:,:,[1,2,0]] #BRG
    elif chance>0.333 and chance <=0.666:
        recon=recon[:,:,[2,1,0]] #BGR
    return im, recon, dmap, t,b,l,r


def tight_crop_depth_joint_masked_depth_withrecon_normald(im,recon, dmap):
    # different tight crop
    msk=(dmap!=0).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    im = im[miny : maxy + 1, minx : maxx + 1, :]
    dmap = dmap[miny : maxy + 1, minx : maxx + 1]
    recon=recon[miny : maxy + 1, minx : maxx + 1, :]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
    dmap = np.pad(dmap, ((s, s), (s, s)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    recon= recon [cy1 : -cy2, cx1 : -cx2, :]
    dmap= dmap [cy1 : -cy2, cx1 : -cx2]
    t=miny-s+cy1
    b=size[0]-maxy-s+cy2
    l=minx-s+cx1
    r=size[1]-maxx-s+cx2
    return im,recon,dmap,t,b,l,r

def tight_crop_rgbp_withrecon(im,recon):
    # different tight crop
    #msk=(dmap<10000).astype(np.uint8)
    msk=((recon[:,:,0]==0)&(recon[:,:,1]==0)&(recon[:,:,2]==0)).astype(np.uint8)
    msk=1-msk
    size=msk.shape
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    im = im[miny : maxy + 1, minx : maxx + 1, :]
    recon=recon[miny : maxy + 1, minx : maxx + 1, :]
    
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1 : -cy2, cx1 : -cx2, :]
    recon= recon[cy1 : -cy2, cx1 : -cx2, :]
    t=miny-s+cy1
    b=size[0]-maxy-s+cy2
    l=minx-s+cx1
    r=size[1]-maxx-s+cx2
    return im,recon,t,b,l,r

def data_aug_full_joint_withrecon_rgbp(im,recon,bg):
    im=im/255.0
    recon=recon/255.0
    bg=bg/255.0
    im,recon,t,b,l,r=tight_crop_rgbp_withrecon(im,recon)
    # change background img
    # msk = fm[:, :, 0] > 0
    #print(im.shape,recon.shape)
    msk=((recon[:,:,0]==0)&(recon[:,:,1]==0)&(recon[:,:,2]==0)).astype(np.uint8)
    msk=1-msk
    msk = np.expand_dims(msk, axis=2)
    # replace bg
    fh, fw, _ = im.shape
    chance=random.random()
    if chance > 0.3:
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[: fh, : fw, :]
    elif chance < 0.3 and chance> 0.2:
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    else:
        bg=np.zeros((fh, fw, 3))
        msk=np.ones((fh, fw, 3))
    im = bg * (1 - msk) + im * msk
    #im = im*msk
    im = color_jitter(im, 0.2, 0.2, 0.6, 0.6)
    #recon = color_jitter(recon, 0.2, 0.2, 0.6, 0.6)
    if chance>0.666:
        recon=recon[:,:,[1,2,0]] #BRG
    elif chance>0.333 and chance <=0.666:
        recon=recon[:,:,[2,1,0]] #BGR
    return im, recon, t,b,l,r