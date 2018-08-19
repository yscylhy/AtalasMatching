# AtalasMatching
This project aims at utilizing the labeled data from Allen Mouse Brain Atalas to train segmetaion networks for diffrent brain structures, such as the dentate gyrus.

## Functions
- **aba_crawler.py:** 
download all the images with label from the Allen Mouse Brain Atalas
- **data_augmentation.py:**
prepare the downloaded data for training
- **dg_seg_train.py:**
the wrapper trains and briefly test the model
- **models.py:**
where the network is defined
