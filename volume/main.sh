#!/bin/bash

# input/output
OUTPUT_CHANNEL=3
BOUND_OUTPUT='False'
WIDTH=1 # boundary width
#DATA_DIR="/home/mil/huang/Dataset/CPR_multiview"
DATA_DIR="/data/ugui0/antonio-t/CPR_multiview_interp2_huang"

# Experiment
EXPERIMENT="Experiment1"
SUB_FOLDER="Res-UNet_CE_3class"

# optimizer
LR_SCHEDULER='StepLR'
MOMENTUM=0.90
GAMMA=0.9
CRITERION='ce'  # cross entropy loss with bound weight
W0=10.0
SIGMA=5.0  # for more sharp boundaries
IGNORE_INDEX='None'
CAL_ZEROGT='False' # whether calculate GT with all pixels equal to zero (only for dice loss)
ALPHA=0.5
OPT='Adam'
WEIGHT='True'
MOD_OUTLINE='False' # modify outline weight to put higher importance on outline
WEIGHT_TYPE='None' # what type of weight to use 'None', 'nlf' or 'mfb'
LR=0.001
STEP_SIZE=10
W_DECAY=0.0005  # almost default setting for segmentation
MPL='False'

# training
SING_GPU_ID=2
ONLY_TEST='False'
NUM_WORKERS=16
BATCH_SIZE=256 # 6 for unet/res_unet
NUM_TRAIN_EPOCHS=100
USE_PRE_TRAIN='False'
PRE_TRAIN_PATH="./Experiment14/2d_res_unet_0.001_0.90_0.9_theta-1.0-0.0_85_200_10_dice_160_96_Adam_rot-True_\
flip-True_w-True_rcp-True_rtrans-False_noise-False_ptr-False_multiview-False_shallow-False_onlyrisk-False_int-32_ds\
-2_alpha-0.5_bc-False_lr-StepLR"
PERCENTILE=100
N_EPOCH_HARDMINING=10
ONLYRISK='False'
CONFIG='config'

# pre-processing/augmentation
R_CENTRAL_CROP='True'
NOISE='False'
FLIP='True'
ROTATION='True'
RANDOM_TRANS='False'
CENTRAL_CROP=192
RESCALE=96
INTERVAL=32
DOWN_SAMPLE=2
MULTI_VIEW='False'
BC_LEARNING='False'

# models
MODEL_TYPE='2d'
MODEL='res_unet_dp'
DROP_OUT=0.0  # drop_out rate for res_unet
THETA=1.0
WITH_SHALLOW_NET='True'

# visualization
DO_PLOT='True'
PLOT_DATA='test'

# create fig_dir to save log file and generated graphs
FIG_DIR="${EXPERIMENT}/${SUB_FOLDER}"

# create $FIG_DIR if it doesn't exist
if [ ! -d "./${FIG_DIR}" ]; then
    mkdir -p ./${FIG_DIR}
fi

LOG="./${FIG_DIR}/train.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo "Logging output to $LOG"

CUDA_VISIBLE_DEVICES=${SING_GPU_ID} python main.py --central_crop ${CENTRAL_CROP} --rescale $RESCALE --output_channel ${OUTPUT_CHANNEL} \
                --num_train_epochs ${NUM_TRAIN_EPOCHS} --w_decay ${W_DECAY} --lr $LR --momentum $MOMENTUM \
                --step_size ${STEP_SIZE} --gamma $GAMMA  --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} \
                --criterion $CRITERION --opt $OPT  --data_dir ${DATA_DIR} --interval ${INTERVAL} --model_type ${MODEL_TYPE}\
                --weight $WEIGHT --only_test ${ONLY_TEST} --rotation $ROTATION --flip $FLIP --r_central_crop ${R_CENTRAL_CROP} \
                --random_trans ${RANDOM_TRANS} --noise $NOISE --use_pre_train ${USE_PRE_TRAIN}  \
                --pre_train_path ${PRE_TRAIN_PATH} --fig_dir ${FIG_DIR} --onlyrisk ${ONLYRISK} \
                --with_shallow_net ${WITH_SHALLOW_NET} --do_plot ${DO_PLOT} --down_sample ${DOWN_SAMPLE}\
                --n_epoch_hardmining ${N_EPOCH_HARDMINING} --percentile ${PERCENTILE} --plot_data ${PLOT_DATA} \
                --multi_view ${MULTI_VIEW} --model ${MODEL} --theta ${THETA} --config ${CONFIG} --bc_learning ${BC_LEARNING} \
                --lr_scheduler ${LR_SCHEDULER} --weight_type ${WEIGHT_TYPE} --mpl ${MPL} --cal_zerogt ${CAL_ZEROGT} \
                --drop_out ${DROP_OUT} --ignore_index ${IGNORE_INDEX}  --w0 ${W0}  --sigma ${SIGMA} --bound_out ${BOUND_OUTPUT} \
                --width ${WIDTH} --mod_outline ${MOD_OUTLINE}