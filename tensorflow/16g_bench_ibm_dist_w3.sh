#/bin/bash

DATA_DIR=/nvme3T/brlee/imagenet/tfrecord
LOG_DIR=/gpfs/gpfs_gl4_16mb/b7p284za/benchmark/tensorflow/output_dist
TIMESTAMP=$(date +%m%d%H%M)

# LOGs
TRAIN_DIR=/gpfs/gpfs_gl4_16mb/b7p284za/benchmark/tensorflow/train_log/resnet101-10e-96b-16G
SUMMARY_VERBOSITY=1
SUMMARIES_STEPS=1000

SAMPLE_SIZE=1281167
NUM_EPOCHS=10
NUM_GPU=16
NUM_GPU_W0=4
NUM_BATCHES=$((${SAMPLE_SIZE} * ${NUM_EPOCHS} / 96 / ${NUM_GPU}))
#NUM_BATCHES=1000
# JOB_NAME : ps or worker
JOB_NAME=$1
# TASK_INDEX : Index of task within the job
TASK_INDEX=$2

mkdir -p ${LOG_DIR}

source /opt/DL/tensorflow/bin/tensorflow-activate
export PYTHONPATH=/gpfs/gpfs_gl4_16mb/b7p284za/anaconda3/lib/python3.6/site-packages
#export GRPC_TRACE=all,-pending_tags
#export GRPC_VERBOSITY=DEBUG

PS_HOST=(129.40.42.115)
W_HOST=(129.40.42.108 129.40.42.114 129.40.42.110 129.40.42.113)
#PS_HOST=(129.40.42.103)
#W_HOST=(129.40.42.103 129.40.42.114)
#W_HOST=(129.40.42.103 129.40.42.114 129.40.42.108 129.40.42.104)
PORT_PS=50000
PORT_WORKER=50001
VARIABLE_UPDATE=distributed_replicated

train() {
    MODEL=$1
    BATCH_SIZE=$2
    JOB_NAME=$3
    TASK_INDEX=$4
    TIMESTAMP=$(date +%m%d%H%M)
    LOG_FILE="${LOG_DIR}/output_tid${TASK_INDEX}_${MODEL}_e${NUM_EPOCHS}_b${BATCH_SIZE}.${TIMESTAMP}_worker_${TASK_INDEX}.log"
    python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --job_name=${JOB_NAME} \
        --ps_hosts="${PS_HOST[0]}:${PORT_PS}" \
        --worker_hosts="${W_HOST[0]}:${PORT_WORKER},${W_HOST[1]}:${PORT_WORKER},${W_HOST[2]}:${PORT_WORKER},${W_HOST[3]}:${PORT_WORKER}" \
        --task_index=${TASK_INDEX} \
        --model=${MODEL} --batch_size=${BATCH_SIZE} --num_batches=${NUM_BATCHES} --num_gpus=${NUM_GPU_W0} \
        --data_name=imagenet --data_dir=${DATA_DIR} --train_dir=${TRAIN_DIR} --variable_update=${VARIABLE_UPDATE} \
        --summary_verbosity=${SUMMARY_VERBOSITY} --save_summaries_steps=${SUMMARIES_STEPS} \
        2>&1 | tee ${LOG_FILE}
}

train resnet101 96 worker 3
#train googlenet 128 worker 0
#train googlenet 128 worker 1
#train googlenet 96 ${JOB_NAME} ${TASK_INDEX}
#train resnet101 32 ${JOB_NAME} ${TASK_INDEX}

