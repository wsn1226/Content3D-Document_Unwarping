import json
from loaders.doc3dwc_loader import doc3dwcLoader
from loaders.doc3dbmnoimgc_loader import doc3dbmnoimgcLoader
from loaders.doc3dboundary_loader import doc3dboundaryLoader
from loaders.doc3dbm_mobilevitfcn_direct_loader_norecon import doc3dbm_mobilevitfcn_direct_loader_norecon
from loaders.doc3dbm_mobilevitfcn_direct_loader import doc3dbm_mobilevitfcn_direct_loader
from loaders.doc3dbm_mobilevitfcn_normal_loader_norecon import doc3dbm_mobilevitfcn_normal_loader_norecon
from loaders.doc3dbm_mobilevitfcn_rgbd_loader_norecon import doc3dbm_mobilevitfcn_rgbd_loader_norecon
from loaders.doc3ddepth_loader import doc3ddepthLoader
from loaders.doc3ddepthfromalb_loader import doc3ddepthfromalb_loader
from loaders.doc3dbm_mobilevitfcn_flow_loader_norecon import doc3dbm_mobilevitfcn_flow_loader_norecon
from loaders.doc3dbm_mobilevitfcn_rgbd_loader_norecon_2 import doc3dbm_mobilevitfcn_rgbd_loader_norecon_2
from loaders.doc3djoint_msked_depth import doc3djoint_masked_depth
from loaders.doc3dfulljoint_msked_depth import doc3djoint_masked_depth_full
from loaders.doc3dfulljoint_msked_depth_withrecon import doc3djoint_masked_depth_full_withrecon
from loaders.doc3dbm_mobilevitfcn_rgbd_loader_reconin_gttrain import doc3dbm_mobilevitfcn_rgbd_loader_norecon_reconin_gttrain
from loaders.doc3dbm_mobilevitfcn_rgbd_flow_loader_norecon import doc3dbm_mobilevitfcn_rgbd_flow_loader_norecon
from loaders.doc3dbm_mobilevitfcn_gd_bm_loader_norecon import doc3dbm_mobilevitfcn_gd_bm_loader_norecon
from loaders.doc3dbm_mobilevitfcn_rgbp_bm_loader_norecon import doc3dbm_mobilevitfcn_rgbp_bm_loader_norecon
from loaders.doc3dbm_mobilevitfcn_rgbp_mand_bm_loader_norecon import doc3dbm_mobilevitfcn_rgbp_mand_bm_loader_norecon
from loaders.doc3dbm_mobilevitfcn_rgbp_sign_bm_loader_norecon import doc3dbm_mobilevitfcn_rgbp_sign_bm_loader_norecon
from loaders.doc3dbm_joint_rgbp_flow_loader_reconin import doc3dbm_joint_rgbp_flow_loader_reconin
from loaders.originalimg_dmap_flow import originalimg_dmap_flow
from loaders.doc3dbm_mobilevitfcn_rgb_diff_loader_norecon import doc3dbm_mobilevitfcn_rgb_diff_loader_norecon
from loaders.doc3dbm_mobilevitfcn_rgb_msk_loader_norecon import doc3dbm_mobilevitfcn_rgb_msk_loader_norecon
from loaders.doc3dbm_rgb_wc import doc3dbm_rgb_wc
from loaders.doc3dbm_rgb_nm import doc3dbm_rgb_nm
from loaders.doc3dbm_nm import doc3dbm_nm
from loaders.doc3dbm_wc import doc3dbm_wc
from loaders.doc3dbm_rgb_posd import doc3dbm_rgb_posd
from loaders.doc3dbm_rgb_absoluted_norecon import doc3dbm_rgb_absoluted_norecon
from loaders.doc3dabs_depthfromalb import doc3dabs_depthfromalb
from loaders.doc3djoint_nm_full import doc3djoint_nm_full

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'doc3dwc':doc3dwcLoader,
        'doc3dbm_nm':doc3dbm_nm,
        'doc3dbm_wc':doc3dbm_wc,
        'doc3djoint_nm_full':doc3djoint_nm_full,
        'doc3dbmnic':doc3dbmnoimgcLoader,
        'doc3dboundary':doc3dboundaryLoader,
        'doc3dbm_rgb_wc':doc3dbm_rgb_wc,
        'doc3dbm_rgb_nm':doc3dbm_rgb_nm,
        'doc3dbm_rgb_absd':doc3dbm_rgb_absoluted_norecon,
        'doc3dabs_depthfromalb':doc3dabs_depthfromalb,
        'doc3dbm_rgb_posd': doc3dbm_rgb_posd,
        'doc3dbm_withrecon':doc3dbm_mobilevitfcn_direct_loader,
        'doc3dbm':doc3dbm_mobilevitfcn_direct_loader_norecon,
        'doc3dbm_flow':doc3dbm_mobilevitfcn_flow_loader_norecon,
        'doc3dbm_rgbd':doc3dbm_mobilevitfcn_rgbd_loader_norecon,
        'doc3dbm_rgbmsk':doc3dbm_mobilevitfcn_rgb_msk_loader_norecon,
        'doc3dbm_rgb_diff':doc3dbm_mobilevitfcn_rgb_diff_loader_norecon,
        'doc3dbm_rgbd_recon_gt':doc3dbm_mobilevitfcn_rgbd_loader_norecon_reconin_gttrain,
        'doc3dbm_rgbd_2':doc3dbm_mobilevitfcn_rgbd_loader_norecon_2,
        'doc3ddepth':doc3ddepthLoader,
        'doc3ddepthfromalb':doc3ddepthfromalb_loader,
        'doc3djoint_masked_depth': doc3djoint_masked_depth,
        'doc3djoint_masked_depth_full': doc3djoint_masked_depth_full,
        'doc3d_rgbd_flow_norecon':doc3dbm_mobilevitfcn_rgbd_flow_loader_norecon,
        'originalimg_dmap_flow':originalimg_dmap_flow,
        'doc3d_rgbp_bm_norecon':doc3dbm_mobilevitfcn_rgbp_bm_loader_norecon,
        'doc3d_joint_rgbp_flow_withrecon':doc3dbm_joint_rgbp_flow_loader_reconin,
        'doc3d_rgbp_sign_bm_norecon':doc3dbm_mobilevitfcn_rgbp_sign_bm_loader_norecon,
        'doc3d_rgbp_mand_bm_norecon':doc3dbm_mobilevitfcn_rgbp_mand_bm_loader_norecon,
        'doc3d_gd_bm_norecon':doc3dbm_mobilevitfcn_gd_bm_loader_norecon,
        'doc3djoint_masked_depth_full_withrecon':doc3djoint_masked_depth_full_withrecon,
    }[name]
