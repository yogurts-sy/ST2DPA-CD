python taba_ktcd_gen.py \
--save_path ./Output_CD/CLCD \
--from_A_to_B False \
--from_B_to_A True \
--target_classes 0 1 3 5 \
--thres_base 50 \
--test_dataset CLCD \
--dataset_root_path /data2/suyou/Codes/datasets/CLCD/test/ \
--config-file ./configs/TABA_CLCD_ViT_B.yaml \
--opts MODEL.WEIGHTS ./Output_Ada/CLCD/model_0029999.pth