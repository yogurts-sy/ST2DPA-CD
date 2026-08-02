import os
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLCD_CATEGORIES = [
    {"color": [128, 128, 0], "id": 1, "name": "bareland"},
    {"color": [0, 0, 128], "id": 2, "name": "building"},
    {"color": [0, 128, 128], "id": 3, "name": "grass"},
    {"color": [255, 255, 0], "id": 4, "name": "pavement"},
    {"color": [0, 255, 0], "id": 5, "name": "trees"},
    {"color": [0, 255, 255], "id": 6, "name": "water"},
]


def _get_CLCD_meta():
    stuff_ids = [k["id"] for k in CLCD_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in CLCD_CATEGORIES]
    stuff_colors = [k["color"] for k in CLCD_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_CLCD(root):
    meta = _get_CLCD_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "imgs", "masks"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"CLCD_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "./Output_Seg/CLCD/sem_seg"
register_CLCD(root)
