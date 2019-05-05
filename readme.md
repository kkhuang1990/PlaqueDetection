# Plaque detection from coronary artery CT images using convolutional neural networks
this is a project for applying deep learning approach to boundary and plaque detection from coronary arteries in cardiac CT images. We have provided easy-used deep learning models such as U-Net, Res-UNet, Tiramisu and DeepLab v2 with both 2D and 3D types. Results are evaluated w.r.t. Dice score (Dice), Absolute volume difference (AVD), Haudorff distance with 95 percentile (HD95) and average symmetric surface distance (ASD) respectively. Besides for medical coronary artery dataset, our codes also work well on the public PROMISE12 dataset. 

## Code composition
main parts of our codes are listed as below:
- BoundDetection

  define functions of training the whole network and corresponding configurations in train.py and main.py respectively. For easy modification of hyper-parameters, define main.sh file with necessary configuration.
- PlaqueSegmentation
  Similar to BoundDetection directory, use train.py and main.py file for training the network and configurations respectively. Difference is that model under this directory is used for plaque detection.
- volume/hybrid/image
  define necessary network structures, dataloader and transforms (image pre-processing and data augmentation) for 3D model, 2.5D model and 2D model respectively. In this work, our novelty is to propose a hybrid Res-UNet structure for taking advantage of spatial connectivity between adjacent slices, meanwhile alleviating computational cost compared with purely 3D network. 
- loss.py
  define loss functions used in this project. Here we propose the snake-constrained weighted Hausdorff distance (WHD) loss with reference to the paper [Locating Objects Without Bounding Boxes](https://arxiv.org/pdf/1806.07564.pdf)
lr_scheduler.py
main.py
main.sh
metric.py
metricsvsnumimages.py
operation_hist.py
operation.py
playground.py
plot_regionprops.py
snake_cp.py
snake.py
train.py
utils.py
vision.py
