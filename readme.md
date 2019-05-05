# Plaque detection from coronary artery CT images using convolutional neural networks
this is a project for applying deep learning approach to boundary and plaque detection from coronary arteries in cardiac CT images. We have provided easy-used deep learning models such as U-Net, Res-UNet, Tiramisu and DeepLab v2 with both 2D and 3D types. Results are evaluated w.r.t. Dice score (Dice), Absolute volume difference (AVD), Haudorff distance with 95 percentile (HD95) and average symmetric surface distance (ASD) respectively. Besides for medical coronary artery dataset, our codes also work well on the public PROMISE12 dataset. 

## Code composition 
main parts of our codes are listed as below:
- BoundDetection
class_weights
configs
datasets
hybrid
image
PlaqueSegmentation
volume
readme.md
__init__.py
loss.py
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
