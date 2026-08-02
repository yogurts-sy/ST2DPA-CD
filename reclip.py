import sys
sys.path.append("..")
import torch
import os
import torch.nn as nn
import numpy as np
import cv2
import open_clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


from matplotlib.patches import Patch
from util.pamr import PAMR
from torchvision import transforms
from open_clip import create_model_and_transforms, tokenizer
from segment_anything import sam_model_registry, SamPredictor
from util.seg_utils import sam_refinement, preprocess_image
from prompts.imagenet_template import openai_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS


VOC_COLORMAP = [[0, 0, 0],
                [255, 255, 255],
                [128, 0, 0],     # building
                [178, 102, 178], # road
                [0, 69, 255],    # river
                [128, 128, 0],   # bareland
                [34, 97, 38],    # forest
                [255, 195, 128], # agriculture
                [0, 255, 36],    # meadow
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


def show_mask(mask, ax, random_color=False, default_color =  [30/255, 144/255, 255/255]):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.concatenate([default_color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color


def show_results(image, save_path, seg_pred, seg_logit, name_list, add_bg=False, vis_thresh = 0.01):
    if type(image) == str:
        image_path = image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cls_pred = F.one_hot(seg_pred.squeeze(0).long(), num_classes=len(name_list) + int(add_bg)).permute(2, 0, 1).float()  # [C, H, W]
    patches = []  # To store legend patches
    plt.figure(figsize=(10, 6)) # ; meadow
    plt.imshow(image)
    for i in range(len(name_list)):
        bool_mask = cls_pred[i].bool() & (seg_logit[i] > vis_thresh)
        cls_color = [c / 255 for c in VOC_COLORMAP[1:][i]] # Skip the first color (black)
        color = show_mask(bool_mask.cpu().numpy(), plt.gca(), default_color=cls_color)
        patch = Patch(color=color, label=str(i) + ' ' + name_list[i])
        patches.append(patch)
    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    image_path = os.path.join(save_path, os.path.basename(image_path))
    plt.savefig(image_path)


def visualize_clip_heatmap_single(image_np,
                                  clip_logits,
                                  class_id=None,
                                  class_name=None,
                                  alpha=0.45,
                                  cmap='jet',
                                  norm='minmax',
                                  clamp_nonneg=False,
                                  save_path=None):

    clip_logits = clip_logits.unsqueeze(0)
    assert clip_logits.ndim == 4 and clip_logits.shape[0] == 1
    H, W = clip_logits.shape[-2], clip_logits.shape[-1]

    with torch.no_grad():
        heat = clip_logits[0, class_id].detach().float()
        if clamp_nonneg:
            heat = torch.clamp_min(heat, 0)

        if norm == 'minmax':
            hmin, hmax = heat.min(), heat.max()
            heat = (heat - hmin) / (hmax - hmin + 1e-6)
        elif norm == 'sigmoid':
            heat = torch.sigmoid(heat)

        heat_np = heat.cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(image_np)
    # plt.imshow(heat_np, cmap=cmap)
    plt.imshow(heat_np, alpha=alpha, cmap=cmap)
    title = f"CLIP similarity heatmap (id={class_id})"
    if class_name is not None:
        title = f"CLIP similarity: {class_name} (id={class_id})"
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)

    plt.show()
    return heat_np


def visualize_clip_heatmap_topk(
    image_np,
    clip_logits,
    class_names=None,
    k=4,
    alpha=0.45,
    cmap='jet',
    norm='minmax',
    clamp_nonneg=False,
    save_dir=None):

    clip_logits = clip_logits.unsqueeze(0)
    assert clip_logits.ndim == 4 and clip_logits.shape[0] == 1
    C = clip_logits.shape[1]

    with torch.no_grad():
        scores = clip_logits[0].mean(dim=(-1, -2))  # [C]
        topk = torch.topk(scores, k=min(k, C), largest=True)
        top_ids = topk.indices.tolist()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    cols = min(k, 4)
    rows = int(np.ceil(len(top_ids) / cols))
    plt.figure(figsize=(4*cols, 4*rows))

    for j, cid in enumerate(top_ids):
        heat = clip_logits[0, cid].detach().float()
        if clamp_nonneg:
            heat = torch.clamp_min(heat, 0)
        if norm == 'minmax':
            hmin, hmax = heat.min(), heat.max()
            heat = (heat - hmin) / (hmax - hmin + 1e-6)
        elif norm == 'sigmoid':
            heat = torch.sigmoid(heat)
        heat_np = heat.cpu().numpy()

        blended = (alpha * plt.cm.get_cmap(cmap)(heat_np)[..., :3] + (1 - alpha) * image_np / 255.0)
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        if save_dir is not None:
            name = class_names[cid] if (class_names and cid < len(class_names)) else f"id_{cid}"
            save_path = os.path.join(save_dir, f"top{j+1}_{name.replace(' ', '_')}.png")
            cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            print(f"[✔] Saved: {save_path}")

        plt.subplot(rows, cols, j + 1)
        plt.imshow(blended)
        plt.axis("off")
        name = class_names[cid] if (class_names and cid < len(class_names)) else f"id={cid}"
        plt.title(f"Top-{j+1}: {name}", fontsize=10)

    plt.tight_layout(pad=0)
    if save_dir is not None:
        summary_path = os.path.join(save_dir, f"top{len(top_ids)}_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"[✔] Summary saved: {summary_path}")
    plt.show()


@MODELS.register_module()
class ReCLIP(BaseSegmentor):
    def __init__(self, clip_type, model_type, vfm_model, name_path, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, beta=1.2, gamma=3.0, slide_stride=112, slide_crop=336, debug=True,
                 sam_refinement=True, sam_model_type='vit_h', pamr_steps=0, pamr_stride=(8, 16),
                 sam_ckpt='./pretrained/sam_vit_h_4b8939.pth',
                 coarse_thresh=0.2, minimal_area=225, sam_mask_coff=0.005, **kwargs):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        if clip_type == 'openai':
            self.clip, _, self.clip_preprocess = create_model_and_transforms(model_type, pretrained='openai', precision='fp16')
        elif clip_type == 'RemoteCLIP':
            self.clip, _, self.clip_preprocess = create_model_and_transforms(model_type, pretrained='./pretrained/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'openclip':
            self.clip, _, self.clip_preprocess = create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            self.clip, _, self.clip_preprocess = create_model_and_transforms(model_type, pretrained='./pretrained/RS5M_ViT-L-14.pt', precision='fp16')
        elif clip_type == 'DGTRSCLIP':
            self.clip, _, self.clip_preprocess = create_model_and_transforms(model_type, pretrained='./pretrained/DGTRS-CLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            self.clip, _, self.clip_preprocess = create_model_and_transforms(model_type, pretrained='./pretrained/vit_large_patch14_224_skyclip_50pct-fc4be2dd.pth"', precision='fp16')
    
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize
        self.clip_stride = int(model_type[-2:])

        self.vfm_model = vfm_model
        # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        from dino.vision_transformer import vit_base
        # self.vfm = vit_base(patch_size=16, num_classes=0)
        # self.vfm.load_state_dict(torch.load('./pretrained/dino_vitbase16_pretrain.pth', map_location="cpu"), strict=True)
        self.vfm = vit_base(patch_size=8, num_classes=0)
        self.vfm.load_state_dict(torch.load('./pretrained/dino_vitbase8_pretrain.pth', map_location="cpu"), strict=True)

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.name_list, _ = get_cls_idx2(name_path)

        self.clip_features = None
        self.clip_logits = None

        query_features = []
        if clip_type == 'openai' or clip_type == 'openclip' or clip_type == 'GeoRSCLIP':
            with torch.no_grad():
                for qw in query_words:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.clip.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.query_features = torch.cat(query_features, dim=0).detach()
        else:
            with torch.no_grad():
                tokenized_text = torch.cat([self.tokenizer(f"a satellite photograph of {c}") for c in query_words]).to(device)
                text_features = self.clip.encode_text(tokenized_text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            self.query_features = text_features
        
        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale                  # 40
        self.prob_thd = prob_thd                        # 0.0
        self.slide_stride = slide_stride                # 112
        self.slide_crop = slide_crop                    # 336
        self.beta = beta                                # 1.2
        self.gamma = gamma                              # 3.0
        self.debug = debug
        self.cross_patch_fusion = 'ave'
        self.query_words = query_words

        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(device=device).eval().half()
        self.sam.prompt_encoder = self.sam.prompt_encoder.float()
        self.sam.mask_decoder = self.sam.mask_decoder.float()

        self.cos_fac = kwargs.get('cos_fac', 1.2)
        self.refine_neg_cos = kwargs.get('refine_neg_cos', True)
        self.sam_iou_thresh = kwargs.get('sam_iou_thresh', 0.9)
        self.save_path = kwargs.get('save_path', '')
        self.sam_predictor = SamPredictor(self.sam)

        self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device) if pamr_steps > 0 else None

        if sam_model_type == 'vit_b':
            self.sam_heads = 12
        elif sam_model_type == 'vit_l':
            self.sam_heads = 16
        elif sam_model_type == 'vit_h':
            self.sam_heads = 16

        if sam_refinement:
            self.sam_refine = True
            self.coarse_thresh = coarse_thresh
            self.minimal_area = minimal_area
            self.sam_mask_coff = sam_mask_coff
        else:
            self.sam_refine = False
        self.kwargs = kwargs

    @torch.no_grad()
    def get_sam_feat(self, tmp_img, stride):
        tmp_h, tmp_w = tmp_img.shape[:2]
        if tmp_h % stride != 0: tmp_h = (tmp_h // stride + 1) * stride
        if tmp_w % stride != 0: tmp_w = (tmp_w // stride + 1) * stride
        tmp_img = cv2.resize(tmp_img, (tmp_w, tmp_h))

        self.sam_predictor.set_image(tmp_img)
        sam_r = 1.0
        # get the feature shape in SAM Encoder
        sam_valid_h, sam_valid_w = sam_r * self.sam_predictor.input_size[0] // stride, sam_r * \
                                   self.sam_predictor.input_size[1] // stride      
        sam_valid_h, sam_valid_w = int(sam_valid_h), int(sam_valid_w)    # 64, 64
        sam_enc_feats = self.sam_predictor.features                      # [1, 256, 64, 64]
        sam_enc_feats = sam_enc_feats[:, :, :sam_valid_h, :sam_valid_w]  # [1, 256, 64, 64]
        sam_hw = int(sam_r * 64)                                         # 64

        sam_attn = self.sam_predictor.model.image_encoder.last_attn      # [12, 64*64, 64*64]
        sam_attn = sam_attn.view(1, self.sam_heads, sam_hw, sam_hw, sam_hw, sam_hw)[:, :, :sam_valid_h, :sam_valid_w,
                   :sam_valid_h, :sam_valid_w]                           # [1, 12, 64, 64, 64, 64]
        sam_attn = sam_attn.flatten(2, 3).flatten(3, 4)                  # [1, 12, 64*64, 64*64]

        sam_v = self.sam_predictor.model.image_encoder.last_v            # [1, 12, 64, 64, 64]
        sam_v = sam_v[:, :, :sam_valid_h, :sam_valid_w, :]               # [1, 12, 64, 64, 64]

        return sam_enc_feats, sam_attn, sam_v, sam_valid_h, sam_valid_w
    
    @torch.no_grad()
    def get_vfm_feat(self, img_batch):
        imgs_norm = [self.norm(self.unnorm(img_batch[i])) for i in range(len(img_batch))]  # replace norm here
        imgs_norm = torch.stack(imgs_norm, dim=0)  # [64, 3, 336, 336]
        imgs_norm = imgs_norm.half()
        feat_out = {}

        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        if self.vfm_model == 'dino':
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

        patch_size = self.vfm.patch_embed.patch_size                     # 16
        if type(patch_size) is tuple: patch_size = patch_size[0]
        feat = self.vfm.get_intermediate_layers(imgs_norm)[0]            # [N, 442, 768]    patches features
        nb_im = feat.shape[0]  # Batch size                              # N
        vfm_h, vfm_w = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size       # 21, 21
        vfm_feats = feat[:, 1:, :].reshape(nb_im, vfm_h, vfm_w, -1).permute(0, 3, 1, 2)                 # batch, c, h, w  -->  [N, 768, 21, 21]

        return vfm_feats, vfm_h, vfm_w  # [N, 768, 21, 21]  21  21

    @torch.no_grad()
    def get_ReCLIP_seg(self, src_img, tmp_img, img_metas):
        
        # get sliding windows
        processed_img = preprocess_image(src_img, self.clip_stride, self.slide_crop)
        clip_whole_h, clip_whole_w = processed_img.shape[-2:]                                                  # 1024, 1024
        clip_feat_h, clip_feat_w = clip_whole_h // self.clip_stride, clip_whole_w // self.clip_stride          # 64, 64
        img_batch, paddings, patch_locs, win_sizes = self.get_windowed_imgs(processed_img, self.clip_stride)   # [64, 3, 336, 336]

        _, sam_attn, sam_v, sam_valid_h, sam_valid_w = self.get_sam_feat(tmp_img, 16)       

        # get vfm features
        vfm_feats, vfm_h, vfm_w = self.get_vfm_feat(img_batch)

        # get clip dense features
        with torch.cuda.amp.autocast():
            clip_features, _ = self.clip.encode_image(img_batch.half(), external_feats=vfm_feats, beta=self.beta, gamma=self.gamma,
                                                paddings=paddings, dst_coords=patch_locs, win_sizes=win_sizes,
                                                dst_vh=clip_feat_h, dst_vw=clip_feat_w, sam_attn=sam_attn, sam_v=sam_v,
                                                cos_fac=self.cos_fac, vfm_token_size=(vfm_h,vfm_w), refine_neg_cos=self.refine_neg_cos)

        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        self.clip_features = clip_features.clone().permute(0, 2, 1).reshape(-1, clip_features.shape[-1], sam_valid_h, sam_valid_w)
        self.clip_features = F.interpolate(self.clip_features, size=img_metas[0]['ori_shape'][:2], mode='bilinear').squeeze(0).permute(1, 2, 0)

        logits = (clip_features) @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], sam_valid_h, sam_valid_w)  # [1, c, 64, 64]
        logits = F.interpolate(logits, size=img_metas[0]['ori_shape'][:2], mode='bilinear')       # [1, c, 1024, 1024]
        self.clip_logits = logits.clone().squeeze(0)                                              # [c, 1024, 1024]

        return logits.float()
    

    def get_windowed_imgs(self, img, patch_size=16):
        stride, crop_size = self.slide_stride, self.slide_crop
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        crop_imgs, paddings, patch_locs = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                assert y1 % patch_size == 0 and x1 % patch_size == 0
                assert y2 % patch_size == 0 and x2 % patch_size == 0
                patch_locs.append(torch.tensor([y1//patch_size, x1//patch_size, y2//patch_size, x2//patch_size]))
                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)
                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                crop_imgs.append(crop_img)
                paddings.append(pad)
        batched_imgs = torch.cat(crop_imgs, dim=0) # [n_patches, 3, h, w]  --->  [64, 3, 336, 336]
        return batched_imgs, paddings, patch_locs, (h_grids, w_grids)

    def predict(self, inputs, tmp_inputs, data_samples, debug_img_path=None):
        if data_samples is not None:
            if self.debug: 
                batch_img_metas = data_samples
            else:
                batch_img_metas = [
                    data_sample.metainfo for data_sample in data_samples
                ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        if self.debug and debug_img_path:
            img_path = debug_img_path
        else:
            img_path = data_samples[0].img_path

        seg_logits = self.get_ReCLIP_seg(inputs, tmp_inputs, batch_img_metas)  # [1, 9, 1024, 1024]

        if self.pamr:
            img_size = batch_img_metas[0]['ori_shape']
            img = nn.functional.interpolate(inputs, size=img_size, mode='bilinear', align_corners=False)
            try:
                seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)
            except RuntimeError as e:
                print(f"Error in PAMR: {e}")

        return self.postprocess_result(seg_logits, tmp_inputs, data_samples, debug_img_path)

    def postprocess_result(self, seg_logits, tmp_img, data_samples, debug_img_path=None):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)  # [0, 1, 2, 3, 4, 5, 6, 6]
            if num_cls != num_queries:

                seg_logits_mixed = torch.zeros((num_cls, seg_logits.shape[1], seg_logits.shape[2])).cuda()

                for ii, idx in enumerate(self.query_idx):
                    seg_logits_mixed[idx] += seg_logits[ii]

                seg_logits = seg_logits_mixed

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            if self.sam_refine:
                assert (data_samples is not None or debug_img_path is not None)
                refined_masks, scores, refined_logits, prompt_boxes = sam_refinement(tmp_img, seg_pred, seg_logits, num_cls,
                                                                                  self.sam_predictor, self.coarse_thresh, self.minimal_area,
                                                                                  self.sam_mask_coff, self.sam_iou_thresh,)
                seg_pred = refined_masks
                seg_logits = refined_logits

            if data_samples is None or self.debug:
                return seg_pred, seg_logits  # [1, size, size], [c, size, size]
            else:

                seg_pred[seg_pred == 1] = 255

                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })

        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')                     
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices

def get_cls_idx2(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('x')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices