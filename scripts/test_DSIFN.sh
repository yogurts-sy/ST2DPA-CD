python taba_ktcd_gen.py \
--save_path ./Output_CD/DSIFN \
--from_B_to_A True \
--target_classes 1 6 \
--thres_base 60 \
--test_dataset DSIFN \
--dataset_root_path /data2/suyou/Codes/datasets/DSIFN-CD512/test/ \
--config-file ./configs/TABA_DSIFN_ViT_B.yaml \
--opts MODEL.WEIGHTS ./Output_Ada/DSIFN/model_0029999.pth