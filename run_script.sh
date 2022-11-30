GPUNUM=8
PROCESSNUM=8
PART=DBX32
JOBNAME=test

srun --partition=${PART} -n${PROCESSNUM} --gres=gpu:${GPUNUM} --job-name=${JOBNAME} --ntasks-per-node=${GPUNUM} --cpus-per-task=5 --quotatype=auto --kill-on-bad-exit=1 \
    python -m torch.distributed.launch train_multi_GPU.py