#!/bin/bash

# input/output
OUTPUT_CHANNEL=3
BOUND_OUTPUT='True'
# For output_channel=2, [inner, outer, innerouter] is available, for output_channel=3, only innerouter is available
BOUND_TYPE='innerouter'
WIDTH=1 # boundary width
#DATA_DIR="/home/mil/huang/Dataset/CPR_multiview"
#DATA_DIR="/data/ugui0/antonio-t/CPR_multiview"
DATA_DIR="/data/ugui0/antonio-t/CPR_multiview_interp2_huang"   # after interpolation with 2 pixels

# Experiment
EXPERIMENT="Experiment12"
SUB_FOLDER="HybridResUNet_int23_0.167_whddb"

# optimizer
LR_SCHEDULER='StepLR'
MOMENTUM=0.90
GAMMA=0.9
CRITERION='whddb' # weighted Hausdorff Distance of double boundaries
## only for Cross Entropy Bound loss which treat inner bounds and outer bounds differently
W1=10.0  # outer bound amplitude
W2=10.0 # inner bound amplitude
SIGMA1=5.0  # outer bound variance
SIGMA2=5.0 # inner bound variance
WHD_ALPHA=4 # only for WHD loss
WHD_BETA=1 # only for WHD loss
WHD_RATIO=0.167 # only for WHD loss
IGNORE_INDEX='None'
CAL_ZEROGT='False' # whether calculate GT with all pixels equal to zero (only for dice loss)
ALPHA=0.5
OPT='Adam'
WEIGHT='False'
MOD_OUTLINE='False' # modify outline weight to put higher importance on outline
WEIGHT_TYPE='None' # what type of weight to use 'None', 'nlf' or 'mfb'
LR=0.001
STEP_SIZE=10
W_DECAY=0.0005
MPL='False'

# training
SING_GPU_ID=5
ONLY_TEST='False'
NUM_WORKERS=16
BATCH_SIZE=32 # 32 for 15 slices and 16 for 31 slices
NUM_TRAIN_EPOCHS=50
USE_PRE_TRAIN='False'
PRE_TRAIN_PATH="./Experiment9/HybridResUNet_int15_ds1_baseline_new"
PERCENTILE=100 # no hard mining
N_EPOCH_HARDMINING=1
ONLY_PLAQUE='False'
CONFIG='config'

# pre-processing/augmentation
R_CENTRAL_CROP='False'
NOISE='False'
FLIP='True'
ROTATION='True'
RANDOM_TRANS='False'
CENTRAL_CROP=192
RESCALE=192
INTERVAL=23
DOWN_SAMPLE=1
MULTI_VIEW='False'
BC_LEARNING='False'

# models
MODEL_TYPE='2.5d'
MODEL='res_unet'
DROP_OUT=0.0  # drop_out rate for res_unet
THETA=1.0
WITH_SHALLOW_NET='True'

# visualization
DO_PLOT='True'
PLOT_DATA='test'

## create fig_dir to save log file and generated graphs
#FIG_DIR="${EXPERIMENT}/${MODEL_TYPE}_${MODEL}_${LR}__${PERCENTILE}_${NUM_TRAIN_EPOCHS}_\
#${STEP_SIZE}_${CRITERION}_${OPT}_r-${ROTATION}_flip-${FLIP}_w-${WEIGHT}_ptr-${USE_PRE_TRAIN}_mv-${MULTI_VIEW}_\
#sl-${WITH_SHALLOW_NET}_\
#lr-${LR_SCHEDULER}_wt-${WEIGHT_TYPE}_o-${OUTPUT_CHANNEL}_b-${BOUND_OUTPUT}_cf-${CONFIG}_dp-${DROP_OUT}_\
#w1-${W1}_w2-${W2}_sg1-${SIGMA1}_sg2-${SIGMA2}_rs-${RESCALE}_wt-${WIDTH}_bt-${BOUND_TYPE}_whda-${WHD_ALPHA}_whdb-${WHD_BETA}"

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
                --pre_train_path ${PRE_TRAIN_PATH} --fig_dir ${FIG_DIR} --only_plaque ${ONLY_PLAQUE} \
                --with_shallow_net ${WITH_SHALLOW_NET} --do_plot ${DO_PLOT} --down_sample ${DOWN_SAMPLE}\
                --n_epoch_hardmining ${N_EPOCH_HARDMINING} --percentile ${PERCENTILE} --plot_data ${PLOT_DATA} \
                --multi_view ${MULTI_VIEW} --model ${MODEL} --theta ${THETA} --config ${CONFIG} --bc_learning ${BC_LEARNING} \
                --lr_scheduler ${LR_SCHEDULER} --weight_type ${WEIGHT_TYPE} --mpl ${MPL} --cal_zerogt ${CAL_ZEROGT} \
                --drop_out ${DROP_OUT} --ignore_index ${IGNORE_INDEX}  --w1 ${W1}  --w2 ${W2}  --sigma1 ${SIGMA1} --sigma2 ${SIGMA2} \
                --bound_out ${BOUND_OUTPUT} --width ${WIDTH} --mod_outline ${MOD_OUTLINE} --bound_type ${BOUND_TYPE} \
                --whd_alpha ${WHD_ALPHA} --whd_beta ${WHD_BETA} --whd_ratio ${WHD_RATIO}