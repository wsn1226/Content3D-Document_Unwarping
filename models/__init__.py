import torchvision.models as models
from models.densenetccnl import *
from models.u2net_depth import U2NET_lite_depth
from models.u2net_depth_vislab8 import U2NET_lite_wc, U2NET_lite_nm
from models.unetnc import *
from models.u2net import *
from models.mobilevitandfcn import *
from models.mobilevitandfcnskip import *
from models.u2net_depth import *
from models.mobilevitandfcnskip_depthfromalb import *
from models.u2net_joint_mobilevit import *
from models.u2net_joint_mobilevit_full import *
from models.stacked_u2net_depth import *
from models.u2net_joint_mobilevit_fullreconin import *
from models.u2net_joint_rgbp_reconin import *
from models.u2net_joint_rgbp_norecon import *
from models.doctr_seg import *
from models.mobilevit_v2_fcnskip import *
from models.mobilevit_v2_fcnskip_revise import *
from models.mobilevitandfcnskip_test import *
from models.u2net_joint_mobilevit_onlydbm import *
from models.mobilevit_6blocks import *
from models.u2net_joint_mobilevit_onlydbm_rgbdiff import *
from models.mobilenetv3 import *
from models.GeoTr import *
from models.extractor import *
from models.position_encoding import *
from models.nm_u2net_joint_mobilevit_full import *

def get_model(name, n_classes=1,img_size=(128,128), filters=64,version=None,in_channels=3, is_batchnorm=True, norm='batch', model_path=None, use_sigmoid=True, layers=3,boundary_model=None,nm_model=None, depth_model=None,prob_model=None, bm_model=None,flow_model=None, d1_model=None, d2_model=None):
    model = _get_model_instance(name)

    if name == 'dnetccnl':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'unetnc':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name=='u2net':
        model = U2NET_lite()
    elif name=='u2net_depth':
        model = U2NET_lite_depth(in_ch=in_channels)
    elif name=='u2net_wc':
        model = U2NET_lite_wc()
    elif name=='u2net_nm':
        model = U2NET_lite_nm()
    elif name=='nm_u2net_joint_mobilevit_full':
        model = nm_u2net_joint_mobilevit_full(boundary_model,nm_model,bm_model)
    elif name=='doctr_seg':
        model = U2NETP(3,1)
    elif name=='mobilevit_sandfcn':
        print("Model: MobileViT Backbone and FCN loaded")
        model = mobilevit_s(img_size)
    elif name=='mobilevit_sandfcn_skip':
        print("Model: MobileViT Backbone and FCN_Skip loaded")
        model = mobilevit_s_skip(img_size)
    elif name=='mobilenetv3_andfcn_skip':
        print("Model: MobileNetV3 and FCN_Skip loaded")
        model = mobilenetv3_large()
    elif name=='mobilevit_xxsandfcn_skip_depthfromalb':
        print("Model: MobileViT Backbone and FCN_Skip loaded for depth from alb")
        model = mobilevit_xxs_skip_depthfromalb(img_size)
    elif name=='mobilevit_sandfcnRGBD':
        print("Model: RGBD MobileViT and FCN is loaded")
        model=mobilevit_s(img_size=img_size,input_channels=4)
    elif name=='mobilevit_sandfcn_skipRGBD':
        print("Model: RGBD MobileViT and FCN_Skip is loaded")
        model=mobilevit_s_skip(img_size=img_size,input_channels=4)
    elif name=='mobilevit_sandfcn_skip_rgb_wc':
        print("Model: RGBD MobileViT and FCN_Skip is loaded")
        model=mobilevit_s_skip(img_size=img_size,input_channels=6)
    elif name=='DocTr':
        print("DocTr is loaded")
        model=GeoTr(6,input_dim=4)
    elif name=='mobilevit_sandfcn_skipRGBD_halfflow':
        print("Model: RGBD MobileViT and FCN_Skip is loaded")
        model=mobilevit_s_skip_halfflow(img_size=img_size,input_channels=4)
    elif name=='mobilevit_sandfcn_skipRGBD_5blocks':
        print("Model: RGBD MobileViT and FCN_Skip is loaded")
        model=mobilevit_s_skip_5vitblocks(img_size=img_size,input_channels=4)
    elif name=='mobilevit_sandfcn_skipRGBD_6blocks':
        print("Model: RGBD MobileViT (6 vit blocks) and FCN_Skip is loaded")
        model=mobilevit_s_skip_6vitblocks(img_size=img_size,input_channels=4)
    elif name=='mobilevit_v2_sandfcn_skipRGBD':
        print("Model: RGBD MobileViT_V2 and FCN_Skip is loaded")
        model=mobilevit_v2_s_skip(img_size=img_size,input_channels=4)
    elif name=='mobilevit_v2_sandfcn_skip_reviseRGBD':
        print("Model: RGBD Revised MobileViT_V2 and FCN_Skip is loaded")
        model=mobilevit_v2_s_skip_revise(img_size=img_size,input_channels=4)
    elif name=='mobilevit_sandfcn_skip_prob':
        print("Model: RGBD MobileViT and FCN_Skip for Probability is loaded")
        model=mobilevit_s_skip(img_size=img_size,input_channels=3,mode='prob',num_classes=1)
    elif name=='mobilevit_sandfcn_skip_prob_sign':
        print("Model: RGBD MobileViT and FCN_Skip for Sign Probability is loaded")
        model=mobilevit_s_skip(img_size=img_size,input_channels=3,num_classes=1)
    elif name=='mobilevit_sandfcn_skipGD':
        print("Model: RGBD MobileViT and FCN_Skip is loaded")
        model=mobilevit_s_skip(img_size=img_size,input_channels=2)
    elif name=='u2netdepth_joint_mobilevit':
        print("Model: Joint u2net and mobilevit is loaded")
        model=u2netdepth_joint_mobilevit(depth_model,bm_model)
    elif name=='u2netdepth_joint_mobilevit_full':
        print("Model: FULL Joint u2net and mobilevit is loaded")
        model=u2netdepth_joint_mobilevit_full(boundary_model, depth_model, bm_model)
    elif name=='u2net_joint_mobilevit_onlydbm':
        print("Model: FULL Joint u2net and mobilevit only Depth and BM is loaded")
        model=u2net_joint_mobilevit_onlydbm(depth_model, bm_model)
    elif name=='u2net_joint_mobilevit_onlydbm_rgbdiff':
        print("Model: FULL Joint u2net and mobilevit only Depth and BM For RGBDiff is loaded")
        model=u2net_joint_mobilevit_onlydbm_rgbdiff(depth_model, bm_model)
    elif name=='u2net_joint_rgbp_reconin':
        print("Model: Joint u2net and mobilevit is loaded for RGBP")
        model=u2net_joint_rgbp_reconin(boundary_model, prob_model, flow_model)
    elif name=='u2net_joint_rgbp_norecon':
        print("Model: Joint u2net and mobilevit is loaded for RGBP NO RECON")
        model=u2net_joint_rgbp_norecon(boundary_model, prob_model, flow_model)      
    elif name=='u2netdepth_joint_mobilevit_fullreconin':
        print("Model: FULL Joint u2net and mobilevit is loaded")
        model=u2netdepth_joint_mobilevit_fullreconin(boundary_model, depth_model, bm_model)
    elif name=='stacked_u2net_depth':
        model=stacked_u2net_depth(d1_model,d2_model)
    else:
        model = model(n_classes=n_classes)
    return model

def _get_model_instance(name):
    try:
        return {
            'dnetccnl': dnetccnl,
            'unetnc': UnetGenerator,
            'u2net': U2NET_lite,
            'u2net_depth': U2NET_lite,
            'mobilevit_sandfcn': MobileViT_FCN,
            'mobilevit_sandfcn_skip': MobileViT_FCN,
            'mobilevit_sandfcnRGBD':MobileViT_FCN,
            'mobilevit_sandfcn_skipRGBD':MobileViT_FCN,
            'mobilevit_xxsandfcn_skip_depthfromalb': MobileViT_FCN,
            'u2netdepth_joint_mobilevit':MobileViT_FCN,
            'u2netdepth_joint_mobilevit_full':MobileViT_FCN,
            'u2netdepth_joint_mobilevit_fullreconin':MobileViT_FCN,
            'stacked_u2net_depth':U2NET_lite,
        }[name]
    except:
        print('Model {} not available'.format(name))
