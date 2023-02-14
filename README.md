# RGBD Document Unwarping

 This is a project supervised by Prof. Kenneth Wong, where we achived SOTA performace in terms of MS-SSIM (0.512578), and Local Distortion (7.581896)

## Contribution and Novelty
 1. We achieved SOTA performance compared with methods with the same pipeline. Specifically, we have improved the SOTA method by 0.32% and 1.93% in terms of MS-SSIM and LD respectively, using only about 1/3 parameters and 79.51% GPU memory.
 2. In document dewarping, we are the first to combine RGB and 3D information to do multimodal learning.
 3. This is the first trial that lets the model distinguish 3D and RGB information in Document Unwarping.

 
## Proposed Pipeline:

![image](https://user-images.githubusercontent.com/78880538/218631901-b915634d-bfb8-4956-863c-a5230dec1855.png)

## Quantitative Comparison

![image](https://user-images.githubusercontent.com/78880538/218633327-3cfb754f-471b-4a95-aa8c-b8741c316cea.png)

## Qualitative Comparison

![image](https://user-images.githubusercontent.com/78880538/218632661-210a89c0-cdd8-41a4-afa1-eb7af234c08c.png)

![image](https://user-images.githubusercontent.com/78880538/218632726-d31af572-be6a-4fec-ad45-e36d6881f226.png)

![image](https://user-images.githubusercontent.com/78880538/218632845-ac982e2d-9700-4a9b-91e9-f3c5e585df49.png)

![image](https://user-images.githubusercontent.com/78880538/218632946-ebfb2abe-ef1d-4af2-a684-f3c85da1664f.png)

![image](https://user-images.githubusercontent.com/78880538/218633125-1f186243-a881-40b3-a6f2-ac68051b13f7.png)

## Ablation Study

![image](https://user-images.githubusercontent.com/78880538/218633402-bd691b10-ee33-4733-ba11-78a9c1947d85.png)

## Training Details
1. First train the three models using ground truth labels 
 For the semantic segmentation task, use cross-entropy loss
 For the depth prediction task, use L1 Loss and the ground truth masked image
 For BM prediction model, we input the ground truth depth and masked image
2. For ground truth training, we trained the BM model for 81 epochs, with batch size = 200, and learning rate = 0.0001. We reduce the learning rate by half when the validation loss doesn’t decrease for 5 epochs continuously. We don’t use auxiliary loss here because we found the performance will be worse if we use the auxiliary loss here.
3. After ground truth training, we do joint training for the 3 models, i.e., the latter 2 models take the previous models’ outputs as their inputs. And we minimize all losses together, including the cross-entropy loss for the semantic segmentation, the L1 Loss for the depth prediction, L1 Loss for the BM prediction, and the auxiliary losses.



