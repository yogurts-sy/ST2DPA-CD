import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_ENABLE_HF_MIRROR"] = "true"
os.environ["HF_HOME"] = "./HF_HOME"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm
sys.path.append("..")

from PIL import Image
from torchvision import transforms
from reclip import ReCLIP
import torch.nn.functional as F
from matplotlib.patches import Patch
from util.visualization import plot
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

VOC_COLORMAP = [[0, 0, 0],
                [255, 255, 255],
                [255, 195, 128],
                [128, 128, 0],
                [34, 97, 38],
                [128, 0, 0],
                [0, 255, 36],
                [0, 69, 255],
                [178, 102, 178],
                [120, 90, 154],
                [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VIS_COLOR_MAP = {
    0: [128, 128, 0],      # bare soil
    1: [0, 0, 128],        # buildings
    2: [0, 128, 128],      # grass
    3: [255, 255, 0],      # pavement
    4: [0, 255, 0],        # trees
}

TAU_PROTOTYPE = { # dense/inst
"bareland": (0.8, 0.8),   
"building": (0.5, 0.5),
"grass":    (0.8, 0.8),
"pavement": (0.8, 0.8),
"tree":     (0.8, 0.8),}

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
    if type(image) == str:
        image_path = image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cls_pred = F.one_hot(seg_pred.squeeze(0).long(), num_classes=len(name_list) + int(add_bg)).permute(2, 0, 1).float()  # [C, H, W]
    patches = []  # To store legend patches
    plt.figure(figsize=(16, 16))
    plt.imshow(image)
    for i in range(len(name_list)):
    
        bool_mask = cls_pred[i].bool() & (seg_logit[i] > vis_thresh)
        cls_color = [c / 255 for c in VOC_COLORMAP[1:][i]] # Skip the first color (black)
        color = show_mask(bool_mask.cpu().numpy(), plt.gca(), default_color=cls_color)
        patch = Patch(color=color, label=str(i) + ' ' + name_list[i])
        patches.append(patch)
    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(pad=0)
    plt.show()

    plt.savefig(save_path)
    plt.close()


def trim_box(bbox):
    x1, y1, x2, y2 = bbox
    x1 = x1 - 10 if x1 - 10 >= 0 else x1
    y1 = y1 - 10 if y1 - 10 >= 0 else y1

    x2 = x2 + 10 if x2 + 10 < 1024 else x2
    y2 = y2 + 10 if y2 + 10 < 1024 else y2

    return x1, y1, x2, y2


def segment_image_bg_trim(image, bg_img, bbox, mask):
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image

    bg_image_array = bg_img.copy()
    bg_image_array[mask, :] = image_array[mask, :]

    image_w_bg = Image.fromarray(bg_image_array)
    x1_ori, y1_ori, x2_ori, y2_ori = bbox
    x1, y1, x2, y2 = trim_box(bbox)
    image_trim_w_bg = image_w_bg.crop((x1, y1, x2, y2))
    image_no_trim_w_bg = image_w_bg.crop((x1_ori, y1_ori, x2_ori, y2_ori))

    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[mask, :] = image_array[mask, :]
    segmented_image = Image.fromarray(segmented_image_array)
    image_no_trim_w_black = Image.new('RGB', image.size, (255, 255, 255))
    transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    image_no_trim_w_black.paste(segmented_image, mask=transparency_mask_image)
    image_no_trim_w_black = image_no_trim_w_black.crop((x1_ori, y1_ori, x2_ori, y2_ori))
    # image_no_trim_w_black.save(f'./output_whu_vis/{img_name}_{AB}_seg_{n}_black.jpg')

    segmented_image_array = np.ones_like(image_array) * 255
    segmented_image_array[mask, :] = image_array[mask, :]
    image_no_trim_w_white = Image.fromarray(segmented_image_array)
    image_no_trim_w_white = image_no_trim_w_white.crop((x1_ori, y1_ori, x2_ori, y2_ori))
    # image_no_trim_w_white.save(f'./output_whu_vis/{img_name}_{AB}_seg_{n}_white.jpg')

    image_trim_ori_bg = image.crop((x1, y1, x2, y2))
    image_no_trim_ori_bg = image.crop((x1_ori, y1_ori, x2_ori, y2_ori))
    # image_no_trim_w_bg.save(f'./output_whu_vis/{img_name}_{AB}_seg_{n}.jpg')
    # segmented_image.save(f'./output_whu_vis/segmented_image_{n}.jpg')

    return image_trim_w_bg, image_trim_ori_bg, image_no_trim_ori_bg, image_no_trim_w_bg, image_no_trim_w_black, image_no_trim_w_white


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


def visualize_seg_mask(seg_mask):
    h, w = seg_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in VIS_COLOR_MAP.items():
        vis[seg_mask == cls_id] = color

    return vis


def crop_image_mining(format_results, img_path, DATASET):
    image = Image.open(img_path)
    bg_image = np.array(Image.open('./data/background/0127.png'))
    ori_w, ori_h = image.size
    annotations = format_results
    mask_h, mask_w = annotations[0]['segmentation'].shape
    if ori_w != mask_w or ori_h != mask_h:
        image = image.resize((mask_w, mask_h))
    cropped_images_w_bg = []
    cropped_boxes = []

    cropped_images_w_bg

    images_no_trim_w_white = []
    images_trim_ori_bg = []
    images_no_trim_w_bg = []
    
    select_mask = []
    real_annotations = []

    for n, mask in enumerate(annotations):
        if DATASET == 'WHU-CD':
            if mask['area'] < 1000 or mask['area'] > 240000: #  
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

        image_trim_w_bg, image_trim_ori_bg, image_no_trim_ori_bg, image_no_trim_w_bg, image_no_trim_w_black, image_no_trim_w_white = segment_image_bg_trim(image, bg_image, bbox, mask['segmentation'])

        images_no_trim_w_white.append(image_no_trim_w_white)
        images_trim_ori_bg.append(image_trim_ori_bg)
        images_no_trim_w_bg.append(image_no_trim_w_bg)

        cropped_boxes.append(bbox)
        select_mask.append(mask['segmentation'])
        real_annotations.append(mask)

    return images_no_trim_w_white, images_trim_ori_bg, images_no_trim_w_bg, cropped_boxes, select_mask, real_annotations


@torch.no_grad()
def retrieve(model, tokenizer, lc_list, preprocess, elements, device) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    stacked_images = torch.stack(preprocessed_images).half()

    tokenized_text = torch.cat([tokenizer(f"a satellite photograph of {c}") for c in lc_list]).to(device)

    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity.cpu().numpy(), image_features


def crop_1024_to_256(save_dir, img_path, img_name, seg_mask, seg_vis):
    imgs_save_path = save_dir + '/sem_seg/imgs'
    masks_save_path = save_dir + '/sem_seg/masks'
    vis_save_path = save_dir + '/sem_seg/masks_vis'

    img = cv2.imread(img_path)

    crop_size = 256
    idx = 0

    for y in range(0, 1024, crop_size):
        for x in range(0, 1024, crop_size):

            crop_img = img[y:y+crop_size, x:x+crop_size]
            crop_seg = seg_mask[y:y+crop_size, x:x+crop_size]
            crop_vis = seg_vis[y:y+crop_size, x:x+crop_size]

            cv2.imwrite(os.path.join(imgs_save_path, f"{img_name}_{idx}.png"), crop_img)
            cv2.imwrite(os.path.join(masks_save_path, f"{img_name}_{idx}.png"), crop_seg.astype(np.uint8))
            cv2.imwrite(os.path.join(vis_save_path, f"{img_name}_{idx}.png"), crop_vis)

            idx += 1

def extract_prediction_from_4_512(model, img, tmp_img):
    seg_pred = torch.zeros((1, 1024, 1024))
    seg_logits = torch.zeros((9, 1024, 1024))

    patch_size = 512
    for i in range(0, 1024, patch_size):
        for j in range(0, 1024, patch_size):
            patch = img[:, :, i:i+patch_size, j:j+patch_size]
            tmp_patch = tmp_img[i:i+patch_size, j:j+patch_size, :]

            seg_pred1, seg_logits1 = model.predict(patch, tmp_patch, data_samples=None, debug_img_path=img_path)

            seg_pred[:, i:i+patch_size, j:j+patch_size] = seg_pred1
            seg_logits[:, i:i+patch_size, j:j+patch_size] = seg_logits1

    return seg_pred, seg_logits

def show_results2(image, seg_pred, seg_logit, name_list, add_bg=False, vis_thresh=0.01, checkpoint_path=None):
    if isinstance(image, str):
        image_path = image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_path = None
        image = np.array(image)

    num_classes = len(name_list) + int(add_bg)
    cls_pred = F.one_hot(seg_pred.squeeze(0).long(), num_classes=num_classes).permute(2, 0, 1).float()  # [C,H,W]

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    ax = plt.gca()
    patches = []

    H, W = seg_pred.shape[-2:]

    for i in range(len(name_list)):
        bool_mask = cls_pred[i].bool() & (seg_logit[i] > vis_thresh)
        if bool_mask.sum() == 0:
            continue

        cls_color = [c / 255 for c in VOC_COLORMAP[1:][i]]
        color = show_mask(bool_mask.cpu().numpy(), ax, default_color=cls_color)
        patches.append(Patch(color=color, label=f"{i}: {name_list[i]}"))

        ys, xs = torch.nonzero(bool_mask, as_tuple=True)
        if len(xs) > 0:
            cx, cy = int(xs.float().mean()), int(ys.float().mean())

            plt.text(cx, cy, name_list[i],
                     color='white', fontsize=10, weight='bold',
                     ha='center', va='center',
                     bbox=dict(facecolor=color, edgecolor='none', alpha=0.7, pad=2))

    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    if checkpoint_path and image_path:
        basename = os.path.basename(checkpoint_path)
        save_dir = os.path.join(os.path.dirname(image_path).replace("images", "output"), basename)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


if __name__ == '__main__':
    CLIP_TYPE = 'RemoteCLIP'
    MODEL_TYPE = 'ViT-L/14'

    SAVE_PATH = './Output_Seg/WHU-CD/' + CLIP_TYPE + '/' + MODEL_TYPE

    # INPUT_PATHs = [
    #         './UCD_Benchmark/WHU-CD-1024/train/A/',
    #         './UCD_Benchmark/WHU-CD-1024/train/B/']
    # SPLITs = ['TRAIN_A', 'TRAIN_B']

    INPUT_PATHs = ['./UCD_Benchmark/WHU-CD-1024/train/B/']
    SPLITs = ['TRAIN_B']

    device = "cuda"
    SAVE_FIGS = False
    SAVE_ALL = True

    name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
                'tree,forest', 'water,river', 'cropland', 'building,roof,house']
    name_list_clip = ['bareland', 'barren', 'grass', 'pavement', 'road',
                'tree', 'forest', 'water', 'river', 'cropland', 'building', 'roof', 'house']
    
    with open('./configs/WHU-CD.txt', 'w') as writers:
        for i in range(len(name_list)):
            if i == len(name_list) - 1:
                writers.write(name_list[i])
            else:
                writers.write(name_list[i] + '\n')
    writers.close()

    os.makedirs(SAVE_PATH, exist_ok=True)

    np.random.seed(42)
    img_size = 1024

    model = ReCLIP(clip_type=CLIP_TYPE, model_type=MODEL_TYPE, vfm_model='dino', name_path='./configs/WHU-CD.txt')
    sam = sam_model_registry['vit_h'](checkpoint='./pretrained/sam_vit_h_4b8939.pth').to(device=device).eval()
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam, 
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.95,
        stability_score_offset=0.9,
        box_nms_thresh=0.3,
        min_mask_region_area=0,
    )

    if SAVE_FIGS:
        os.makedirs(SAVE_PATH + '/insts/building', exist_ok=True)
        os.makedirs(SAVE_PATH + '/insts/grass', exist_ok=True)
        os.makedirs(SAVE_PATH + '/insts/bareland', exist_ok=True)
        os.makedirs(SAVE_PATH + '/insts/tree', exist_ok=True)
        os.makedirs(SAVE_PATH + '/insts/pavement', exist_ok=True)
    if SAVE_ALL:
        os.makedirs(SAVE_PATH + '/sem_seg/imgs', exist_ok=True)
        os.makedirs(SAVE_PATH + '/sem_seg/masks', exist_ok=True)
        os.makedirs(SAVE_PATH + '/sem_seg/masks_vis', exist_ok=True)

    for idx in range(len(INPUT_PATHs)):

        INPUT_PATH = INPUT_PATHs[idx]
        split = SPLITs[idx]

        img_names = []
        for f in os.listdir(INPUT_PATH):
            if str(f).endswith('.png') or str(f).endswith('.jpg'):
                img_names.append(f)

        img_names = sorted(img_names)

        for img_name in tqdm(img_names):
            img_path = os.path.join(INPUT_PATH, img_name)

            img_name_no_ext = os.path.splitext(img_name)[0]

            img = Image.open(img_path)
            img_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            ])(img)

            img_tensor = img_tensor.unsqueeze(0).to('cuda')

            tmp_img = cv2.imread(img_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)

            seg_pred, seg_logits = extract_prediction_from_4_512(model, img_tensor, tmp_img)
            show_results(img_path, seg_pred, seg_logits, name_list, os.path.join(SAVE_PATH, f'{split}_{img_name}'), vis_thresh=0.01)

            everything_results = mask_generator.generate(tmp_img)

            seg_pred = seg_pred.squeeze(0).cpu().numpy()

            SEG_MASK = np.zeros_like(seg_pred)
            SEG_MASK[SEG_MASK == 0] = 255

            bld_mask = (seg_pred == 8)
            grass_mask = (seg_pred == 2) | (seg_pred == 7)
            bareland_mask = (seg_pred == 1)
            tree_mask = (seg_pred == 5)
            pavement_mask = (seg_pred == 3) | (seg_pred == 4)


            i_n_t_w_white, images_trim_ori_bg, images_no_trim_w_bg, cropped_boxes, select_mask, annotations = crop_image_mining(everything_results, img_path, 'WHU-CD')

            similarity, image_features = retrieve(model.clip, model.tokenizer, name_list_clip, model.clip_preprocess, images_no_trim_w_bg, device)

            # plot(image=np.array(Image.open(img_path)), annotations=np.array(select_mask), output_path=os.path.join(SAVE_PATH + '/sem_seg/masks_vis', f'any_{os.path.basename(img_path)}'), withContours=True)

            if SAVE_ALL:
                SEG_MASK[bareland_mask] = 0
                SEG_MASK[grass_mask] = 2
                SEG_MASK[pavement_mask] = 3
                SEG_MASK[tree_mask] = 4

            for n in range(len(annotations)):
                anno = annotations[n]
                prob = similarity[n]
                i_white = i_n_t_w_white[n]
                i_t_o_b = images_trim_ori_bg[n]
                i_n_w_b = images_no_trim_w_bg[n]

                # i_n_w_b.save('i_n_w_b.png')
                # img_np = cv2.imread(img_path)
                # x1, y1, x2, y2 = cropped_boxes[n]
                # cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.imwrite("proposal.png", img_np)

                bld_ratio = np.sum(bld_mask[anno['segmentation']]) / anno['area']
                grass_ratio = np.sum(grass_mask[anno['segmentation']]) / anno['area']
                bareland_ratio = np.sum(bareland_mask[anno['segmentation']]) / anno['area']
                tree_ratio = np.sum(tree_mask[anno['segmentation']]) / anno['area']
                pavement_ratio = np.sum(pavement_mask[anno['segmentation']]) / anno['area']

                bld_prob = prob[10] + prob[11] + prob[12]
                grass_prob = prob[2] + prob[9]
                bareland_prob = prob[0] + prob[1]
                tree_prob = prob[5] + prob[6]
                pavement_prob = prob[3] + prob[4]

                if bareland_ratio > TAU_PROTOTYPE['bareland'][0]:
                    if SAVE_FIGS:
                        i_white.save(SAVE_PATH + '/bareland' + f'/{split}_{img_name.replace(".png", "")}_ba_{str(n)}.png')
                    SEG_MASK[anno['segmentation']] = 0
                    continue
                if bld_ratio > TAU_PROTOTYPE['building'][0]:
                    b = f'{bld_prob:.2f}'
                    SEG_MASK[anno['segmentation']] = 1
                    if bld_prob < TAU_PROTOTYPE['building'][1]:
                        continue
                    if SAVE_FIGS:
                        i_white.save(SAVE_PATH + '/building' + f'/{split}_{img_name.replace(".png", "")}_b_{b}_{str(n)}.png')
                    continue
                if grass_ratio > TAU_PROTOTYPE['grass'][0]:
                    if SAVE_FIGS:
                        i_white.save(SAVE_PATH + '/grass' + f'/{split}_{img_name.replace(".png", "")}_g_{str(n)}.png')
                    SEG_MASK[anno['segmentation']] = 2
                    continue
                if pavement_ratio > TAU_PROTOTYPE['pavement'][0]:
                    if SAVE_FIGS:
                        i_white.save(SAVE_PATH + '/pavement' + f'/{split}_{img_name.replace(".png", "")}_p_{str(n)}.png')
                    SEG_MASK[anno['segmentation']] = 3
                    continue
                if tree_ratio > TAU_PROTOTYPE['tree'][0]:
                    if SAVE_FIGS:
                        i_white.save(SAVE_PATH + '/tree' + f'/{split}_{img_name.replace(".png", "")}_t_{str(n)}.png')
                    SEG_MASK[anno['segmentation']] = 4
                    continue

            SEG_MASK[SEG_MASK == 255] = 3 # set to pavement
            # cv2.imwrite(os.path.join(SAVE_PATH, 'SEM_SEG', f'{split}_{img_name}'), SEG_MASK.astype(np.uint8))

            vis_mask_rgb = visualize_seg_mask(SEG_MASK.astype(np.uint8))
            vis_mask_bgr = cv2.cvtColor(vis_mask_rgb, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(SAVE_PATH, 'SEM_SEG', f'vis_{split}_{img_name}'), vis_mask_bgr)

            crop_1024_to_256(SAVE_PATH, img_path, f'{split}_{img_name_no_ext}', SEG_MASK.astype(np.uint8), vis_mask_bgr)
