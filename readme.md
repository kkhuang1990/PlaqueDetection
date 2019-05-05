# Plaque detection from coronary artery CT images using convolutional neural networks
this is a project for applying deep learning approach to boundary and plaque detection from coronary arteries in cardiac CT images. We have provided easy-used deep learning models such as U-Net, Res-UNet, Tiramisu and DeepLab v2 with both 2D and 3D types. Results are evaluated w.r.t. Dice score (Dice), Absolute volume difference (AVD), Haudorff distance with 95 percentile (HD95) and average symmetric surface distance (ASD) respectively. Besides for medical coronary artery dataset, our codes also work well on the public PROMISE12 dataset. 

## Code composition
main parts of our codes are listed as below:
- BoundDetection
  - define functions of training the whole network and corresponding configurations in train.py and main.py respectively. For easy modification of hyper-parameters, define main.sh file with necessary configuration.
- PlaqueSegmentation
  - Similar to BoundDetection directory, use train.py and main.py file for training the network and configurations respectively. Difference is that model under this directory is used for plaque detection.
- volume/hybrid/image
  - define necessary network structures, dataloader and transforms (image pre-processing and data augmentation) for 3D model, 2.5D model and 2D model respectively. In this work, our novelty is to propose a hybrid Res-UNet structure for taking advantage of spatial connectivity between adjacent slices, meanwhile alleviating computational cost compared with purely 3D network. 
- loss.py
  - define loss functions used in this project. We propose the snake-constrained weighted Hausdorff distance (WHD) loss with reference to the paper [Locating Objects Without Bounding Boxes](https://arxiv.org/pdf/1806.07564.pdf). As a modification, we apply snake constraint to the original WHD loss to further ensure that a closed artery boundary can be obtained during training. 
- metric.py
  - define commonly used metrics for evaluating results of boundary detection and plaque segmentation, such as Dice, AVD, HD95 and ASD as mentioned above.
- snake.py
  - define snake constraint applied to WHD loss. 
- vision.py
  - define functions for results visualization such as saving samples stack into pdf and show the prediction results in video. 
  
## How to run
running this project is very simple. For plaque detection, change to directory PlaqueSegmentation directory, modify the parameters setting in main.sh and run the main.sh file with ./main.sh. Please notice that datasets used for train, validation and test should be defined first (with both input image and annotation). For boundary detection, similar operations to plaque detection are necessary.  

## sample results
### results on Coronary Artery dataset
<img src="http://i.imgur.com/Jjwsc.jpg" alt="エビフライトライアングル" title="サンプル">
