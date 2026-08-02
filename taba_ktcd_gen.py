import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_ENABLE_HF_MIRROR"] = "true"
os.environ["HF_HOME"] = "/data2/suyou/Codes/UDA-CD/HF_HOME"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pprint
import sys
import argparse
import re
import math
import multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append("..")

from PIL import Image
from torchvision import transforms
from matplotlib.patches import Patch
from open_clip import tokenizer
from util.visualization import plot
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from util.utils import Evaluator, intersectionAndUnion
from detectron2.config import get_cfg
from TABA import add_TABA_config
from predictor import VisualizationDemo
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

np.random.seed(42)

VOC_COLORMAP = [[0, 0, 0],
                [64, 0, 0],
                [64, 0, 0],
                [64, 0, 0],
                [64, 0, 0],
                [64, 0, 0],
                [64, 0, 0],
                [64, 0, 0],
                [178, 102, 178], # road
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

color_map = np.array([[[0, 0, 0], [0, 0, 185]], [[0, 178, 0], [255, 255, 255]]])

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_TABA_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

from sklearn.decomposition import PCA
def visualize_sam_feature(
    sam_feat,
    save_path="./sam_feature_pca.png",
):
    """
    Visualize SAM feature using PCA.

    Args:
        sam_feat: Tensor, [1, 256, 64, 64]
        save_path: output image path
    """

    assert sam_feat.ndim == 4, \
        f"Expected [B, C, H, W], but got {sam_feat.shape}"

    B, C, H, W = sam_feat.shape
    assert B == 1, "Current visualization only supports batch size = 1"

    # [1, 256, 64, 64] -> [64, 64, 256]
    feat = sam_feat[0].detach().float().cpu()
    feat = feat.permute(1, 2, 0).numpy()

    # [64, 64, 256] -> [4096, 256]
    feat_flat = feat.reshape(-1, C)

    # PCA: 256 -> 3
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat_flat)

    # Normalize each PCA component
    feat_pca = (
        feat_pca - feat_pca.min(axis=0, keepdims=True)
    ) / (
        feat_pca.max(axis=0, keepdims=True)
        - feat_pca.min(axis=0, keepdims=True)
        + 1e-8
    )

    # [4096, 3] -> [64, 64, 3]
    feat_rgb = feat_pca.reshape(H, W, 3)

    # Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(feat_rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)

    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )

    plt.close()

    print(f"SAM feature visualization saved to: {save_path}")

    return feat_rgb

def show_mask(mask, ax, random_color=False, default_color =  [30/255, 144/255, 255/255]):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.concatenate([default_color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color

def show_results(image, seg_pred, seg_logit, name_list, save_path, add_bg=False, vis_thresh = 0.01):
    seg_pred = torch.from_numpy(seg_pred).cuda()
    seg_logit = torch.from_numpy(seg_logit).cuda()
    if type(image) == str:
        image_path = image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cls_pred = F.one_hot(seg_pred.long(), num_classes=len(name_list) + int(add_bg)).permute(2, 0, 1).float()  # [C, H, W]
    patches = []  # To store legend patches
    plt.figure(figsize=(16, 16))
    plt.imshow(image)
    for i in range(len(name_list)):
        if i != 0 and i != 1 and i != 3 and i != 5:
            continue
    
        bool_mask = cls_pred[i].bool() & (seg_logit[i] > vis_thresh)
        cls_color = [c / 255 for c in VOC_COLORMAP[1:][i]] # Skip the first color (black)
        color = show_mask(bool_mask.cpu().numpy(), plt.gca(), default_color=cls_color)
        patch = Patch(color=color, label=str(i) + ' ' + name_list[i])
        patches.append(patch)
    plt.axis('off')
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(pad=0)
    plt.show()

    plt.savefig(save_path)
    plt.close()

VIS_COLOR_MAP = {
    0: [128, 128, 0],      # bare soil
    1: [0, 0, 128],        # buildings
    2: [0, 128, 128],      # grass
    3: [255, 255, 0],      # pavement
    4: [0, 255, 0],        # trees
}

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
])

TAU_CD = { # dense
"LEVIR-CD": 0.8,
"WHU-CD": 0.5,
"CLCD": 0.5,
"DSIFN": 0.5,
"SECOND": 0.5
}

SECOND_CLASS_MAPPING = {
    'background': 0,
    'grass': 1,
    'ground': 2,
    'tree': 3,
    'water': 4,
    'building': 5,
    'playground': 6,
}
SECOND_CLASSNAME_MAPPING = [
    "ground",
    "building",
    "grass",
    "pavement",
    "tree",
    "water",
    "playground"
]

def cal_cos_smilarity(d1_p1, d2_p1):
    cos = torch.nn.CosineSimilarity(dim=0)

    return cos(d1_p1, d2_p1).item()

def cal_cos_smilarity_float(prev_inte_hire_features, curr_inte_hire_features):

    prev_inte_hire_features = prev_inte_hire_features.squeeze(0)
    curr_inte_hire_features = curr_inte_hire_features.squeeze(0)

    multi_dotsum_prev_curr = torch.sum(prev_inte_hire_features * curr_inte_hire_features, dim=0, keepdim=True)

    prev_norm = torch.norm(prev_inte_hire_features, dim=0, keepdim=True)
    curr_norm = torch.norm(curr_inte_hire_features, dim=0, keepdim=True)

    multi_dis = prev_norm * curr_norm

    cos_float01 = (multi_dotsum_prev_curr / multi_dis) * 0.5 + 0.5
    
    return cos_float01

def extract_features(image: np.ndarray, model, processor, model_config: dict, device: str) -> torch.Tensor:
    """
    Process the input image and extract features using the specified model.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        model: Pre-trained model for feature extraction.
        processor: Pre-processing function for images.
        model_config (dict): Configuration dictionary containing model type, feature dimension, and patch size.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Extracted feature tensor.
    """
    model_type = model_config['model_type']
    feature_dim = model_config['feature_dim']
    patch_size = model_config['patch_size']

    if model_type == 'DINO':
        processed_image = processor(image).unsqueeze(0).to(device)
        h, w = processed_image.shape[-2:]
        features = model.get_intermediate_layers(processed_image)[0]
        features = features[:, 1:].permute(0, 2, 1).view(1, feature_dim, h // patch_size, w // patch_size)
    elif model_type == 'DINOv2':
        processed_image = processor(images=image, return_tensors="pt").to(device)
        h, w = processed_image['pixel_values'].shape[-2:]
        features = model(**processed_image)
        features = features.last_hidden_state[:, 1:].permute(0, 2, 1).view(1, feature_dim, h // patch_size, w // patch_size)
    else:
        raise ValueError("Unsupported model type. Use 'DINO' or 'DINOv2'.")

    return features

def KTCD(change_proposals, **kwargs):

    dm_cons_cosdis = kwargs.get('dm_cons_cosdis')
    dst_v1 = kwargs.get('dst_v1')
    dst_v2 = kwargs.get('dst_v2')
    bcd_mask = kwargs.get('bcd_mask')
    thres_mask = kwargs.get('thres_mask')
    thres_base = kwargs.get('thres_base')

    for p in change_proposals:
        
        p_mask = p['anno']['segmentation']

        p_mask_idx_one = np.where(p_mask, 1, 0).astype(np.float32)
        p_multi_cosdis_one = np.multiply(dm_cons_cosdis, p_mask_idx_one) * 255
        p_multi_cosdis_avg_one = np.mean(p_multi_cosdis_one[p_mask])

        cos_score = (p_multi_cosdis_avg_one - thres_base) / thres_base if p_multi_cosdis_avg_one > thres_base else 0

        d1_p1 = torch.mean(dst_v1[:, p_mask].permute(1, 0), dim=0).squeeze(0)
        d2_p1 = torch.mean(dst_v2[:, p_mask].permute(1, 0), dim=0).squeeze(0)

        sim = cal_cos_smilarity(d1_p1, d2_p1)
        p_similarity = (sim + 1) / 2
        p_similarity_score = 5. * np.log(1 / p_similarity)

        if p_similarity_score + cos_score < 0:
            p_similarity_score = 0.1
        else:
            p_similarity_score = p_similarity_score + cos_score

        p_mask_idx = np.where(p_mask, p_similarity_score, 0).astype(np.float32)
        p_multi_cosdis = np.multiply(dm_cons_cosdis, p_mask_idx) * 255
        p_multi_cosdis_avg = np.mean(p_multi_cosdis[p_mask])

        try:
            p_multi_cosdis_thres = math.ceil(thres_base / p_similarity_score)
        except Exception:
            p_multi_cosdis_thres = thres_base

        p_multi_cosdis[p_mask] = p_multi_cosdis_avg

        un_bcd_mask = (bcd_mask.astype(np.uint8) == 0)
        un_bcd_mask = un_bcd_mask & p_mask
        bcd_mask[un_bcd_mask] = p_multi_cosdis[un_bcd_mask]

        un_thres_mask = (thres_mask.astype(np.uint8) == 0)
        un_thres_mask = un_thres_mask & p_mask
        thres_mask[un_thres_mask] = p_multi_cosdis_thres

    return bcd_mask, thres_mask

def PWD(change_proposals, **kwargs):
    img1 = kwargs.get('img1')
    img2 = kwargs.get('img2')
    bcd_mask = kwargs.get('bcd_mask')
    thres_mask = kwargs.get('thres_mask')
    diff = torch.abs(img1 - img2)
    diff = diff.mean(dim=0, keepdim=True)

    pred = (diff > 45/255.).float().squeeze(0).cpu().numpy()

    for p in change_proposals:

        p_mask = p['anno']['segmentation']

        un_bcd_mask = (bcd_mask.astype(np.uint8) == 0)
        un_bcd_mask = un_bcd_mask & p_mask
        bcd_mask[un_bcd_mask] = pred[un_bcd_mask]

    return bcd_mask, thres_mask

def SD(change_proposals, **kwargs):
    target_mask1 = kwargs.get('target_mask1')
    target_mask2 = kwargs.get('target_mask2')
    bcd_mask = kwargs.get('bcd_mask')
    thres_mask = kwargs.get('thres_mask')

    diff = target_mask1 != target_mask2
    pred = np.zeros_like(target_mask1)
    pred[diff] = 1.0

    for p in change_proposals:

        p_mask = p['anno']['segmentation']

        un_bcd_mask = (bcd_mask.astype(np.uint8) == 0)
        un_bcd_mask = un_bcd_mask & p_mask
        bcd_mask[un_bcd_mask] = pred[un_bcd_mask]

    return bcd_mask, thres_mask

def LM(change_proposals, **kwargs):
    sam_enc_feats1 = kwargs.get('sam_enc_feats1') # [256, 1024, 1024]
    sam_enc_feats2 = kwargs.get('sam_enc_feats2') # [256, 1024, 1024]
    bcd_mask = kwargs.get('bcd_mask')
    thres_mask = kwargs.get('thres_mask')

    for p in change_proposals:

        p_mask = p['anno']['segmentation']

        un_bcd_mask = (bcd_mask.astype(np.uint8) == 0)
        un_bcd_mask = un_bcd_mask & p_mask
        t1_mask_embed = torch.mean(sam_enc_feats1[:, p_mask], dim=-1)
        t2_mask_embed = torch.mean(sam_enc_feats2[:, p_mask], dim=-1)
        score = -F.cosine_similarity(t1_mask_embed, t2_mask_embed, dim=0)
        bcd_mask[un_bcd_mask] = score.cpu().numpy()

        thres_mask[un_bcd_mask] = math.cos(155 / 180 * math.pi)

    return bcd_mask, thres_mask

def LMD(change_proposals, **kwargs):
    sam_enc_feats1 = kwargs.get('dst_v1') # [768, 1024, 1024]
    sam_enc_feats2 = kwargs.get('dst_v2') # [768, 1024, 1024]
    bcd_mask = kwargs.get('bcd_mask')
    thres_mask = kwargs.get('thres_mask')

    for p in change_proposals:

        p_mask = p['anno']['segmentation']

        un_bcd_mask = (bcd_mask.astype(np.uint8) == 0)
        un_bcd_mask = un_bcd_mask & p_mask
        t1_mask_embed = torch.mean(sam_enc_feats1[:, p_mask], dim=-1)
        t2_mask_embed = torch.mean(sam_enc_feats2[:, p_mask], dim=-1)
        score = -F.cosine_similarity(t1_mask_embed, t2_mask_embed, dim=0)
        bcd_mask[un_bcd_mask] = score.cpu().numpy()

        thres_mask[un_bcd_mask] = math.cos(145 / 180 * math.pi)

    return bcd_mask, thres_mask

def instance_threshold_for_proposal(change_proposals, **kwargs):
    change_head = kwargs.get('change_head')
    
    if change_head == 'KTCD':
        bcd_mask, thres_mask = KTCD(change_proposals, **kwargs)

    elif change_head == 'PWD':
        bcd_mask, thres_mask = PWD(change_proposals, **kwargs)

    elif change_head == 'SD':
        bcd_mask, thres_mask = SD(change_proposals, **kwargs)

    elif change_head == 'LM':
        bcd_mask, thres_mask = LM(change_proposals, **kwargs)

    elif change_head == 'LMD':
        bcd_mask, thres_mask = LMD(change_proposals, **kwargs)

    return bcd_mask, thres_mask

def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            # Merge multiple bounding boxes into one.
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
        h = y2 - y1
        w = x2 - x1
    return [x1, y1, x2, y2]

def get_parser(taba_config_file, taba_weights):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default=taba_config_file,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", taba_weights],
        nargs=argparse.REMAINDER,
    )
    return parser

def extract_annotations(format_results, DATASET):
    annotations = format_results

    cropped_boxes = []
    real_annotations = []

    for n, mask in enumerate(annotations):
        if DATASET == 'WHU-CD':
            if mask['area'] < 1000 or mask['area'] > 240000:
                continue
        elif DATASET == 'LEVIR-CD':
            if mask['area'] < 500 or mask['area'] > 160000:
                continue
        elif DATASET == 'DSIFN':
            if mask['area'] < 500 or mask['area'] > 100000:
                continue
        elif DATASET == 'SECOND':
            if mask['area'] < 1000:
                continue
        elif DATASET == 'BANDON':
            if mask['area'] < 1000:
                continue

        bbox = get_bbox_from_mask(mask['segmentation'])

        cropped_boxes.append(bbox)
        real_annotations.append(mask)

    return cropped_boxes, real_annotations

@torch.no_grad()
def get_sam_feat(sam_predictor, tmp_img):
    sam_predictor.set_image(tmp_img)
    sam_r = 1.0
    stride = 16
    # get the feature shape in SAM Encoder
    sam_valid_h, sam_valid_w = sam_r * sam_predictor.input_size[0] // stride, sam_r * \
                                sam_predictor.input_size[1] // stride      
    sam_valid_h, sam_valid_w = int(sam_valid_h), int(sam_valid_w)    # 64, 64
    sam_enc_feats = sam_predictor.features                           # [1, 256, 64, 64]
    sam_enc_feats = sam_enc_feats[:, :, :sam_valid_h, :sam_valid_w]  # [1, 256, 64, 64]

    return sam_enc_feats

@torch.no_grad()
def extract_dense_prediction(identifier_model, img, img_name, clsn):
    seg_mask = np.zeros((img.shape[0], img.shape[1]))
    sem_segs = np.zeros((clsn, img.shape[0], img.shape[1]))
    dst_v = torch.zeros((768, img.shape[0], img.shape[1])).cuda()

    patch_size = 256
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[0], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            # cv2.imwrite(os.path.join('./vis_cost_out_WHU/imgs', f"{img_name}_{i}_{j}.png"), patch)

            predictions, visualized_output = identifier_model.run_on_image(patch, f"{img_name}_{i}_{j}")

            sem_seg = predictions['sem_seg']
            sem_segs[:, i:i+patch_size, j:j+patch_size] = sem_seg.cpu().numpy()
            sem_seg = sem_seg.softmax(0)

            dst_v1 = predictions['dst_v']
            dst_v1 = F.interpolate(dst_v1, (patch_size, patch_size), mode='bilinear', align_corners=True).squeeze(0)
            dst_v[:, i:i+patch_size, j:j+patch_size] = dst_v1

            sem_pred = sem_seg.argmax(0, keepdim=True).squeeze(0).cpu().numpy()

            seg_mask[i:i+patch_size, j:j+patch_size] = sem_pred

    return seg_mask, dst_v, sem_segs


def remove_overlap(DATASET, change_proposals):
    real_change_proposals = []
    real_change_proposals_annotations = []
    overlap_thresh = 0.3

    if DATASET in ['LEVIR-CD']:
        anno_mask = np.zeros((1024, 1024), dtype=bool)

        change_proposals = sorted(change_proposals, key=lambda x: x['anno']['area'])
        for i in range(len(change_proposals)):
            anno = change_proposals[i]
            inner_ratio = np.sum(anno_mask[anno['anno']['segmentation']]) / anno['anno']['area']
            if inner_ratio > overlap_thresh:
                continue

            anno_mask[anno['anno']['segmentation']] = True
            real_change_proposals.append(anno)
            real_change_proposals_annotations.append(anno['anno']['segmentation'])
    else:

        change_proposals = sorted(change_proposals, key=lambda x: x['anno']['area'])
        real_change_proposals = change_proposals
        for i in range(len(change_proposals)):
            anno = change_proposals[i]
            real_change_proposals_annotations.append(anno['anno']['segmentation'])

    return real_change_proposals, real_change_proposals_annotations

def visualize_seg_mask(seg_mask):
    h, w = seg_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in VIS_COLOR_MAP.items():
        vis[seg_mask == cls_id] = color

    return vis

def visualization(save_path, sem_pred):
    vis_mask_rgb = visualize_seg_mask(sem_pred.astype(np.uint8))
    vis_mask_bgr = cv2.cvtColor(vis_mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_mask_bgr)

def extract_change_proposals(target_mask, everything_results, AB, **kwargs):
    change_proposals = []
    IMG_PATH = kwargs.get('img_path')
    SAVE_DIR = kwargs.get('save_path')
    DATASET = kwargs.get('dataset')

    cropped_boxes, annotations = extract_annotations(everything_results, DATASET)
    # plot(image=np.array(Image.open(IMG_PATH)), annotations=np.array(select_mask1), output_path=os.path.join(SAVE_DIR, AB, f'any_{AB}_{os.path.basename(IMG_PATH)}'), withContours=True)

    for n in range(len(annotations)):
        anno = annotations[n]
        target_ratio = np.sum(target_mask[anno['segmentation']]) / anno['area']

        # img1_np = cv2.imread(IMG_PATH)
        # x1, y1, x2, y2 = cropped_boxes[n]
        # cv2.rectangle(img1_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imwrite("proposal.png", img1_np)

        if DATASET == 'WHU-CD':

            if target_ratio >= TAU_CD[DATASET]:
                change_proposals.append({'n':n, 'anno': anno})

        elif DATASET == 'LEVIR-CD':

            if target_ratio >= TAU_CD[DATASET]:
                change_proposals.append({'n':n, 'anno': anno})

        elif DATASET == 'CLCD':

            if target_ratio >= TAU_CD[DATASET]:
                change_proposals.append({'n':n, 'anno': anno})

        elif DATASET == 'DSIFN':

            if target_ratio >= TAU_CD[DATASET]:
                change_proposals.append({'n':n, 'anno': anno})

        elif DATASET == 'SECOND':

            if target_ratio >= TAU_CD[DATASET]:
                change_proposals.append({'n':n, 'anno': anno})

    real_change_proposals, real_change_proposals_annotations = remove_overlap(DATASET, change_proposals)
    # plot(image=np.array(Image.open(IMG_PATH)), annotations=np.array(real_change_proposals_annotations),
    #     output_path=os.path.join(SAVE_DIR, AB, f'target_{AB}_{os.path.basename(IMG_PATH)}'), withContours=True)

    return real_change_proposals

def unsupervised_change_detection(identifier_model, img1_path, img2_path, label_path, evaluator_test, **kwargs):

    DATASET = kwargs.get('dataset')
    SAVE_DIR = kwargs.get('save_path')
    TARGET_CLASSES = kwargs.get('target_classes')
    THRES_BASE = kwargs.get('thres_base')
    FROM_A_TO_B = kwargs.get('from_A_to_B')
    FROM_B_TO_A = kwargs.get('from_B_to_A')
    LOGGER = kwargs.get('logger')
    CHANGE_HEAD = kwargs.get('change_head')
    IMG_NAME = kwargs.get('img_name')
    CLSN = kwargs.get('clsn')

    img1 = read_image(img1_path, format="BGR")
    img2 = read_image(img2_path, format="BGR")

    img1_t = TRANSFORMS(Image.open(img1_path).convert('RGB')).cuda()
    img2_t = TRANSFORMS(Image.open(img2_path).convert('RGB')).cuda()

    tmp_img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    tmp_img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    target_mask1 = np.zeros((img1.shape[0], img1.shape[1]))
    target_mask2 = np.zeros((img2.shape[0], img2.shape[1]))

    if DATASET == 'SECOND':
        
        class_id = SECOND_CLASS_MAPPING[SECOND_CLASSNAME_MAPPING[TARGET_CLASSES[0]]]
        mask1 = np.array(Image.open(label_path), dtype=np.uint8)
        mask2 = np.array(Image.open(label_path.replace("T1", "T2")), dtype=np.uint8)
        mask  = ((mask1 == class_id) | (mask2 == class_id)).astype(np.uint8) * 255
    
    else:

        mask = np.array(Image.open(label_path).convert('L'), dtype=np.uint8)
        
    # import time
    # start_time = time.time()
    sam_enc_feats1, sam_enc_feats2, dm_cons_cosdis = None, None, None

    # 1. Extract latent change proposals

    sam_enc_feats1 = get_sam_feat(mask_generator.predictor, tmp_img1)
    sam_enc_feats2 = get_sam_feat(mask_generator.predictor, tmp_img2)

    seg_pred1, dst_v1, sem_segs1 = extract_dense_prediction(identifier_model, img1, os.path.splitext(IMG_NAME)[0], CLSN)
    seg_pred2, dst_v2, sem_segs2 = extract_dense_prediction(identifier_model, img2, os.path.splitext(IMG_NAME)[0], CLSN)    

    for target_class in TARGET_CLASSES:
        target_mask1[seg_pred1 == target_class] = 1

    target_mask1 = target_mask1.astype(np.uint8)

    for target_class in TARGET_CLASSES:
        target_mask2[seg_pred2 == target_class] = 1

    target_mask2 = target_mask2.astype(np.uint8)

    if FROM_A_TO_B:
        everything_results1 = mask_generator.generate(tmp_img1)

    if FROM_B_TO_A:
        everything_results2 = mask_generator.generate(tmp_img2)

    # 2. Calculate cosine similarity

    dm_cons_cossim = cal_cos_smilarity_float(sam_enc_feats1, sam_enc_feats2)
    dm_cons_cossim = F.interpolate(dm_cons_cossim.unsqueeze(0), (seg_pred1.shape[0], seg_pred1.shape[0]), mode='bilinear', align_corners=True)
    dm_cons_cossim = dm_cons_cossim.squeeze(0).squeeze(0).cpu().numpy()

    dm_cons_cosdis = 1 - dm_cons_cossim
    dm_cons_cosdis = np.clip(dm_cons_cosdis, 0, 1).astype(np.float32)

    if FROM_A_TO_B:
        change_proposals1 = extract_change_proposals(target_mask1, everything_results1, 'A', save_path=SAVE_DIR, dataset=DATASET, target_classes=TARGET_CLASSES, img_path=img1_path)

    if FROM_B_TO_A:
        change_proposals2 = extract_change_proposals(target_mask2, everything_results2, 'B', save_path=SAVE_DIR, dataset=DATASET, target_classes=TARGET_CLASSES, img_path=img2_path)

    bcd_mask1 = np.zeros((seg_pred1.shape[0], seg_pred1.shape[1]))
    thres_mask1 = np.zeros((seg_pred1.shape[0], seg_pred1.shape[1]))

    bcd_mask2 = np.zeros((seg_pred1.shape[0], seg_pred1.shape[1]))
    thres_mask2 = np.zeros((seg_pred1.shape[0], seg_pred1.shape[1]))

    sam_enc_feats1 = F.interpolate(sam_enc_feats1, (seg_pred1.shape[0], seg_pred1.shape[0]), mode='bilinear', align_corners=True).squeeze(0)
    sam_enc_feats2 = F.interpolate(sam_enc_feats2, (seg_pred1.shape[0], seg_pred1.shape[0]), mode='bilinear', align_corners=True).squeeze(0)

    if FROM_A_TO_B:
        target_mask_other = target_mask2.copy()

        bcd_mask1, thres_mask1 = instance_threshold_for_proposal(change_proposals1, dm_cons_cosdis=dm_cons_cosdis, target_mask1=target_mask1, 
                    target_mask2=target_mask_other, dst_v1=dst_v1, dst_v2=dst_v2, bcd_mask=bcd_mask1, thres_mask=thres_mask1, thres_base=THRES_BASE, 
                    AB='A', change_head=CHANGE_HEAD, img1=img1_t, img2=img2_t, sam_enc_feats1=sam_enc_feats1, sam_enc_feats2=sam_enc_feats2)

    if FROM_B_TO_A:
        target_mask_other = target_mask1.copy()

        bcd_mask2, thres_mask2 = instance_threshold_for_proposal(change_proposals2, dm_cons_cosdis=dm_cons_cosdis, target_mask1=target_mask_other, 
                    target_mask2=target_mask2, dst_v1=dst_v1, dst_v2=dst_v2, bcd_mask=bcd_mask2, thres_mask=thres_mask2, thres_base=THRES_BASE, 
                    AB='B', change_head=CHANGE_HEAD, img1=img1_t, img2=img2_t, sam_enc_feats1=sam_enc_feats1, sam_enc_feats2=sam_enc_feats2)

    bcd_mask_int8 = np.where((bcd_mask1 > thres_mask1) | (bcd_mask2 > thres_mask2), 255, 0).astype(np.uint8)

    # end_time = time.time()
    # inference_time = (end_time - start_time) * 1000
    # LOGGER.info(f"Inference time: {inference_time:.3f} ms")

    # cv2.imwrite(os.path.join(SAVE_DIR, 'CM', os.path.basename(img2_path)), bcd_mask_int8)    

    df_int8 = bcd_mask_int8
    df_int8 = df_int8 / 255

    if DATASET in ['DSIFN']:
        mask = mask
    else:
        mask = mask / 255

    mask = mask[np.newaxis, :]
    df_int8 = df_int8[np.newaxis, :]

    evaluator_test.add_batch(mask.astype('int'), df_int8.astype('int'))

    intersection, union, target = \
            intersectionAndUnion(mask.astype('int'), df_int8.astype('int'), 2, 255)

    iou = intersection / (union + 1e-10)

    pred_rgb_out = np.array([color_map[l][p] for l, p in zip(mask.astype('int').reshape(-1), df_int8.astype('int').reshape(-1))])
    pred_rgb_out = pred_rgb_out.reshape((mask.shape[1], mask.shape[2], 3)).astype(np.uint8)
    # cv2.imwrite(os.path.join(SAVE_DIR, 'CM', 'color_mask_' + os.path.basename(img2_path)), pred_rgb_out)

    LOGGER.info('{:} IoU: {:1f} '.format(img1_path, iou[1]))
    with open(os.path.join(SAVE_DIR, 'IoU.txt'), 'a') as txt_file:
        txt_file.write('{:} IoU: {:1f}'.format(os.path.basename(img1_path), iou[1]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UDA-CD')
    parser.add_argument('--save_path', type=str, default='/data2/suyou/Codes2/UDA-CD/Output_KTCD', help='path to save evaluation results')
    parser.add_argument('--from_A_to_B', type=bool, default=True, help='inference from time t1 to time t2')
    parser.add_argument('--from_B_to_A', type=bool, default=True, help='inference from time t2 to time t1')
    parser.add_argument('--target_classes', type=int, nargs='+', default=[1], help='the list of target classes')
    parser.add_argument('--thres_base', type=int, default=70, help='change vector threshold')
    parser.add_argument('--test_dataset', type=str, default='WHU-CD', help='evaluation dataset: WHU-CD, LEVIR-CD, DSIFN, CLCD')
    parser.add_argument('--change_head', type=str, default='KTCD', help='change head: KTCD, PWD, SD, LM, LMD')
    parser.add_argument('--sam_root_path', type=str, default='./pretrained/sam_vit_h_4b8939.pth', help='segment anything model root path')
    parser.add_argument('--dataset_root_path', type=str, default='/data2/suyou/Codes/datasets/WHU-CD-1024/test/', help='evaluation dataset root path')
    parser.add_argument("--config-file", default='./configs/TABA_WHU_ViT_B.yaml', metavar="FILE", help="path to config file",)
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=["MODEL.WEIGHTS", './Output_Ada/Abalation/Attention/WHU_TABA_attn/model_0029999.pth'], nargs=argparse.REMAINDER,)

    args = parser.parse_args()

    DATASET = args.test_dataset
    INPUT_DIR = args.dataset_root_path
    TARGET_CLASSES = args.target_classes
    THRES_BASE = args.thres_base
    FROM_A_TO_B = args.from_A_to_B
    FROM_B_TO_A = args.from_B_to_A
    SAVE_DIR = args.save_path
    SAM_ROOT_PATH = args.sam_root_path
    CHANGE_HEAD = args.change_head

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'A'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'B'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'CM'), exist_ok=True)
 
    evaluator_test = Evaluator(num_class=2)
    
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    all_args = {**vars(args)}
    with open(os.path.join(SAVE_DIR, 'IoU.txt'), 'a') as txt_file:
        txt_file.write('{}\n\n'.format(pprint.pformat(all_args)))

    cfg = setup_cfg(args)

    sam = sam_model_registry['vit_h'](checkpoint=SAM_ROOT_PATH).to(device="cuda").eval()

    if DATASET == 'DSIFN':
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,
            points_per_batch=64,
            pred_iou_thresh=0.0,
            stability_score_thresh=0.5,
            box_nms_thresh=0.7,
        )

    elif DATASET == 'SECOND':
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,
            points_per_batch=224,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.95,
            stability_score_offset=0.9,
            box_nms_thresh=0.3,
            min_mask_region_area=0,
        )

    else:

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,
            points_per_batch=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            box_nms_thresh=0.3,
        )

    model = VisualizationDemo(cfg)

    img_names = []
    if DATASET in ['WHU-CD', 'LEVIR-CD']:

        T1_NAME = 'A'
        T2_NAME = 'B'
        MASK_NAME = 'label'
        IMG_SIZE = 1024
        CLSN = 5

    elif DATASET == 'DSIFN':

        T1_NAME = 't1'
        T2_NAME = 't2'
        MASK_NAME = 'mask'
        IMG_SIZE = 512
        CLSN = 7

    elif DATASET == 'CLCD':

        T1_NAME = 'time1'
        T2_NAME = 'time2'
        MASK_NAME = 'label'
        IMG_SIZE = 512
        CLSN = 6

    for f in os.listdir(os.path.join(INPUT_DIR, T1_NAME)):
        if str(f).endswith(".png") or str(f).endswith(".jpg"):
            img_names.append(f)

    if DATASET == 'WHU-CD':
        img_names = sorted(img_names)
    elif DATASET == 'LEVIR-CD':
        img_names = sorted(img_names, key=lambda x: int(re.search(r'_(\d+)', x).group(1)))
    elif DATASET == 'DSIFN':
        img_names = sorted(img_names, key=lambda x: int(x.rstrip('.jpg')))
    elif DATASET == 'CLCD':
        img_names = sorted(img_names, key=lambda x: int(x.rstrip('.png')))
    elif DATASET == 'SECOND':
        img_names = sorted(img_names, key=lambda x: int(x.rstrip('.png')))

    for img_name in tqdm(img_names, desc="Processing", unit="iteration"):

        # if img_name != '0022.png':
        #     continue

        img1_path = os.path.join(INPUT_DIR, T1_NAME, img_name)
        img2_path = os.path.join(INPUT_DIR, T2_NAME, img_name)

        if DATASET == 'DSIFN':
            label_path = os.path.join(INPUT_DIR, MASK_NAME, img_name.replace(".jpg", '.tif'))
        else:
            label_path = os.path.join(INPUT_DIR, MASK_NAME, img_name)

        unsupervised_change_detection(model, img1_path, img2_path, label_path, evaluator_test, dataset=DATASET, save_path=SAVE_DIR, 
                                        target_classes=TARGET_CLASSES, thres_base=THRES_BASE, from_A_to_B=FROM_A_TO_B, from_B_to_A=FROM_B_TO_A, 
                                        logger=logger, change_head=CHANGE_HEAD, img_name=img_name, clsn=CLSN)

    IoU = evaluator_test.Intersection_over_Union()[1]
    ACC = evaluator_test.OA()[1]
    Pre = evaluator_test.Precision()[1]
    Recall = evaluator_test.Recall()[1]
    F1 = evaluator_test.F1()[1]
    Kappa = evaluator_test.Kappa()[1]

    logger.info('\n***** Evaluation ***** >>>> IoU: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f, OA: %.2f, Kappa: %.4f\n' % ( IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa))

    with open(os.path.join(SAVE_DIR, 'IoU.txt'), 'a') as txt_file:
        txt_file.write('\n***** Evaluation ***** >>>> IoU: {:.2f}, F1: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, OA: {:.2f}, Kappa: {:.4f}\n'.format(IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa) + '\n')