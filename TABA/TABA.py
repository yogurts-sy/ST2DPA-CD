# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import os
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from einops import rearrange

def count_trainable_params(model):
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Trainable Params: {trainable_params / 1e6:.3f} M")
    print(f"Total Params: {total_params / 1e6:.3f} M")
    print(f"Trainable Ratio: {100 * trainable_params / total_params:.4f}%")

    return trainable_params

def visualize_corr(corr_1, files_name, save_prefix='./output_vis_cost/'):

    # get picture path
    file_name = os.path.basename(files_name)
    file_base_name = os.path.splitext(file_name)[0]

    cv2_img = Image.open(files_name)
    cv2_img = cv2.cvtColor(np.array(cv2_img), cv2.COLOR_RGB2BGR)

    for b in range(corr_1.shape[1]):
        H, W = cv2_img.shape[:2]

        vis_1 = corr_1.squeeze(1)[:, b, :, :]
        vis_1 = (vis_1 - vis_1.min()) / (vis_1.max() - vis_1.min())
        vis = F.interpolate(vis_1.unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0)
        vis = vis.permute(1, 2, 0).cpu().detach().numpy()
        vis = (vis * 255).astype('uint8')
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        vis = cv2_img * 0.4 + vis * 0.6

        save_dir = os.path.join(save_prefix, file_base_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, '{}_{}.jpg'.format(file_name,b))
        cv2.imwrite(save_path, vis)

def visualize_corr2(corr_1, save_prefix='./output_vis_cost/', file_name='img.png'):

    for b in range(corr_1.shape[1]):
        if b != 4:
            continue
        H, W = 256, 256

        vis_1 = corr_1.squeeze(1)[:, b, :, :]
        vis_1 = (vis_1 - vis_1.min()) / (vis_1.max() - vis_1.min())
        vis = F.interpolate(vis_1.unsqueeze(0), (H, W), mode='bilinear', align_corners=False).squeeze(0)
        vis = vis.permute(1, 2, 0).cpu().detach().numpy()
        vis = (vis * 255).astype('uint8')
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)

        save_path = os.path.join(save_prefix, '{}-{}.jpg'.format(file_name, b))
        cv2.imwrite(save_path, vis)

@META_ARCH_REGISTRY.register()
class TABA(nn.Module):
    @configurable
    
    
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        clip_pretrained: str,
        clip_decod_guid_dim: list,
        text_guidance_dim:int,
        use_remote_clip:bool,
        clip_pretrained_remote:str,
        use_vanilla_dino:bool,
        dino_finetune:bool,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.use_remote_clip = use_remote_clip

        self.clip_decod_dim = clip_decod_guid_dim
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        # Vanilla CLIP
        self.clip_finetune = clip_finetune
        clip_finetune = "attention"
        self.set_vanilla_clip_finetune(clip_finetune)

        count_trainable_params(
            self.sem_seg_head.predictor.clip_model
        )

        self.sliding_window = sliding_window
        self.dino_resolution = (384, 384)
        if clip_pretrained == "ViT-B/16": 
            self.clip_resolution = (384, 384)
            self.proj_dim = 768
        else: 
            self.clip_resolution = (336, 336)
            self.proj_dim = 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        
        # Remote CLIP
        if self.use_remote_clip:
            self.set_remote_clip_frozen()

            self.clip_pretrained_remote = clip_pretrained_remote

            if clip_pretrained_remote == "ViT-B/32": 
                self.proj_dim_remote = 768
            else:
                self.proj_dim_remote = 1024
            self.clip_resolution_remote = (768, 768)
            self.upsample1_remote = nn.ConvTranspose2d(self.proj_dim_remote, 256, kernel_size=2, stride=2)
            self.upsample2_remote = nn.ConvTranspose2d(self.proj_dim_remote, 128, kernel_size=4, stride=4)
            self.layer_indexes_remote = [3, 7] if clip_pretrained_remote == "ViT-B/32" else [7, 15] 
            self.layers_remote = []
            for l in self.layer_indexes_remote:
                self.sem_seg_head.predictor.clip_model_remote.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers_remote.append(o))

        # Vanilla DINO
        self.use_vanilla_dino = use_vanilla_dino
        self.dino_finetune = dino_finetune
    
        if self.use_vanilla_dino:
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

            from dino.vision_transformer import vit_base

            self.vfm = vit_base(patch_size=8, num_classes=0)
            self.vfm.load_state_dict(torch.load('./pretrained/dino_vitbase8_pretrain.pth', map_location="cpu"), strict=True)
            
            if self.dino_finetune:
                self.vfm = self.set_vanilla_dino_finetune(self.vfm, "attention")
            else:
                for p in self.vfm.parameters():
                    p.requires_grad = False
                self.vfm.eval().to(self.device)
            
            self.dino_decod_proj1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0)
            self.dino_decod_proj2 = nn.ConvTranspose2d(in_channels=768, out_channels=128, kernel_size=2, stride=2)
            self.dino_down_sample = nn.Conv2d(in_channels=768, out_channels=text_guidance_dim, kernel_size=2, stride=2, padding=0)

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "clip_decod_guid_dim":cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_DIMS,
            "text_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM,
            "clip_pretrained_remote":cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_REMOTE,
            "use_remote_clip" : cfg.MODEL.SEM_SEG_HEAD.USE_REMOTE_CLIP,
            "use_vanilla_dino": cfg.MODEL.SEM_SEG_HEAD.USE_VANILLA_DINO,
            "dino_finetune": cfg.MODEL.SEM_SEG_HEAD.DINO_FINETUNE,
        }
    
    def set_vanilla_dino_finetune(self, dino, dino_ft):
        for name, params in dino.named_parameters():
            if dino_ft == "attention":
                if "attn.qkv.weight" in name:
                    if "blocks.3" in name or "blocks.7" in name or "blocks.11" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif "pos_embed" in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            elif dino_ft == "full":
                params.requires_grad = True
            else:
                params.requires_grad = False
        return dino

    def set_vanilla_clip_finetune(self, clip_finetune):
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False # Trainable Params: 30.686 M Trainable Ratio: 20.5093% 12132M
                elif clip_finetune == "attention":
                    if "attn" in name:
                        params.requires_grad = True if ("q_proj" in name or "k_proj" in name) else False     # 1. Q.K.  12664M     Trainable Params: 20.447 M  Total Params: 149.621 M Trainable Ratio: 13.6660%
                        # params.requires_grad = True if ("q_proj" in name or "v_proj" in name) else False     # 2. Q.V.  12648M    Trainable Params: 20.447 M  Total Params: 149.621 M  Trainable Ratio: 13.6660%
                        # params.requires_grad = True if ("q_proj" in name or "out_proj" in name) else False   # 3. Q.O.  12082M    Trainable Params: 20.463 M  Total Params: 149.621 M  Trainable Ratio: 13.6763%
                        # params.requires_grad = True if ("q_proj" in name or "k_proj" in name or "v_proj" in name) else False   # 4. Q.K.V.  12822M  Trainable Params: 30.671 M  Trainable Ratio: 20.4991%
                        # params.requires_grad = True if ("q_proj" in name or "k_proj" in name or "out_proj" in name) else False   # 5. Q.K.O.  12132M  Trainable Params: 30.686 M  Trainable Ratio: 20.5093%
                        # params.requires_grad = True if ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name) else False   # 6. Q.K.V.O.  12170M  Trainable Params: 40.910 M  Trainable Ratio: 27.3423%
                        # params.requires_grad = True # Attn  40.956 M 27.3731%   12172M

                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True  # 15174M 122.883 M 100%
                else:
                    params.requires_grad = False
            
            else:
                params.requires_grad = False

    def set_vanilla_clip_fronzen(self, clip_finetune):
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():  # 8194M Trainable Params: 0.000 M  Trainable Ratio: 0.0000% 
            params.requires_grad = False

    def set_remote_clip_frozen(self):

        for name, params in self.sem_seg_head.predictor.clip_model_remote.named_parameters():
            params.requires_grad = False

    @property
    def device(self):
        return self.pixel_mean.device
    # @profile(precision=4,stream=open('./log.txt','w+',encoding="utf-8"))
    def forward(self, batched_inputs):

        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """

        if self.training:
            # images_shape: 384*384
            images = [x["image"].to(self.device) for x in batched_inputs]
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
            self.layers = []
            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )

            if self.use_remote_clip:
                self.layers_remote = []
                clip_images_resized_remote = F.interpolate(clip_images.tensor, size=self.clip_resolution_remote, mode='bilinear', align_corners=False, )
            if self.use_vanilla_dino:
                dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )

        else:
            if self.sliding_window:
                with torch.no_grad():
                    kernel = 384
                    overlap = 0.333
                    out_res = [640, 640]
                    images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
                    stride = int(kernel * (1 - overlap))
                    unfold = nn.Unfold(kernel_size=kernel, stride=stride)
                    fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

                    image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
                    image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
                    global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
                    image = torch.cat((image, global_image), dim=0)

                    images = (image - self.pixel_mean) / self.pixel_std
                    clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
                    self.layers = []
                    clip_images_resized = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )

                    if self.use_remote_clip:
                        self.layers_remote = []
                        clip_images_resized_remote = F.interpolate(clip_images, size=self.clip_resolution_remote, mode='bilinear', align_corners=False, )
                    if self.use_vanilla_dino:
                        dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, ) 
            else:
                with torch.no_grad():
                    images = [x["image"].to(self.device) for x in batched_inputs]
                    clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
                    clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
                    self.layers = []
                    clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )

                    if self.use_remote_clip:
                        self.layers_remote = []
                        clip_images_resized_remote = F.interpolate(clip_images.tensor, size=self.clip_resolution_remote, mode='bilinear', align_corners=False, )
                    if self.use_vanilla_dino:
                        dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )
        
        # Vanilla CLIP
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        clip_image_features = clip_features[:, 1:, :]
        res3 = rearrange(clip_image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4) if self.upsample1 is not None else None
        res5 = self.upsample2(res5) if self.upsample2 is not None else None
        clip_features_guidance = {'res5': res5, 'res4': res4, 'res3': res3,}

        clip_features_input = clip_features

        # Remote CLIP
        if self.use_remote_clip:
            H_c = 24 if self.clip_pretrained_remote == 'ViT-B/32' else 54
            _ = self.sem_seg_head.predictor.clip_model_remote.encode_image(clip_images_resized_remote, dense=True)
            res4 = rearrange(self.layers_remote[0][1:, :, :], "(H W) B C -> B C H W", H=H_c)
            res5 = rearrange(self.layers_remote[1][1:, :, :], "(H W) B C -> B C H W", H=H_c)
            res4 = self.upsample1_remote(res4) if self.upsample1_remote is not None else None
            res5 = self.upsample2_remote(res5) if self.upsample2_remote is not None else None
            if self.clip_pretrained_remote == 'ViT-L/14':
                res4 = F.interpolate(res4, size=(48, 48), mode='bilinear', align_corners=False, )
                res5 = F.interpolate(res5, size=(96, 96), mode='bilinear', align_corners=False, )
            clip_features_guidance_remote = {'res5': res5, 'res4': res4,}
        else:
            clip_features_guidance_remote = None

        # Vanilla DINO
        if self.use_vanilla_dino:
            dino_feat = self.vfm.get_intermediate_layers(dino_images_resized, n=12)
            dino_patch_feat_last_unfold = rearrange(dino_feat[-1][:,1:,:], "B (H W) C -> B C H W", H=48)

            dino_feat_input = self.dino_down_sample(dino_patch_feat_last_unfold)
            dino_feat_L4 = rearrange(dino_feat[3][:,1:,:], "B (H W) C -> B C H W", H=48)
            dino_feat_L8 = rearrange(dino_feat[7][:,1:,:], "B (H W) C -> B C H W", H=48)

            dino_feat_L4_proj = self.dino_decod_proj1(dino_feat_L4)
            dino_feat_L8_proj = self.dino_decod_proj2(dino_feat_L8)
            dino_feat_guidance = [dino_feat_L4_proj, dino_feat_L8_proj]
        else:
            dino_feat_input, dino_feat_guidance = None, None

        files_name = [f'img_{str(i)}' for i, x in enumerate(batched_inputs)]
        outputs = self.sem_seg_head(files_name,
        clip_features_input, 
        dino_feat_input,
        clip_features_guidance, 
        clip_features_guidance_remote,
        dino_feat_guidance)

        # if you want to visualize the corr map
        # visualize_corr2(outputs, save_prefix='./vis_cost_out_WHU/corrs', file_name=f'corr_{files_name[0]}')

        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)

            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            # targets[mask] = targets[mask] - 1

            outputs = outputs.permute(0, 2, 3, 1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot

            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses
            
        else:
            if self.sliding_window:
                with torch.no_grad():
                    outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
                    outputs = outputs.sigmoid()
                    
                    global_output = outputs[-1:]
                    global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
                    outputs = outputs[:-1]
                    outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
                    outputs = (outputs + global_output) / 2.

                    height = batched_inputs[0].get("height", out_res[0])
                    width = batched_inputs[0].get("width", out_res[1])
                    output = sem_seg_postprocess(outputs[0], out_res, height, width)
                    return [{'sem_seg': output}]
            else:
                with torch.no_grad():
                    outputs = outputs.sigmoid()
                    image_size = clip_images.image_sizes[0]
                    height = batched_inputs[0].get("height", image_size[0])
                    width = batched_inputs[0].get("width", image_size[1])

                    output = sem_seg_postprocess(outputs[0], image_size, height, width)
                    processed_results = [{'sem_seg': output, 'dst_v': dino_patch_feat_last_unfold}]
                    return processed_results        
        