#!/usr/bin/env bash
#SBATCH -p lowprio
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, AL
#SBATCH --output=/data/%u/master_thesis/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/data/%u/master_thesis/jobs/%j.err # where to store error messages
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=A100:1

USERNAME=$USER
PROJECT_NAME=master_thesis
DIRECTORY=/data/${USERNAME}/${PROJECT_NAME}
MAMBA_ENVIRONMENT=dmiise

mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

module load mamba gpu cuda/12.4
nvidia-smi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-24.3.0-0-snzyeuqww5oky4utpxyepf7ymqes4mme/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-24.3.0-0-snzyeuqww5oky4utpxyepf7ymqes4mme/etc/profile.d/conda.sh" ]; then
        . "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-24.3.0-0-snzyeuqww5oky4utpxyepf7ymqes4mme/etc/profile.d/conda.sh"
    else
        export PATH="/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-24.3.0-0-snzyeuqww5oky4utpxyepf7ymqes4mme/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-24.3.0-0-snzyeuqww5oky4utpxyepf7ymqes4mme/etc/profile.d/mamba.sh" ]; then
    . "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-24.3.0-0-snzyeuqww5oky4utpxyepf7ymqes4mme/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

mamba activate ${MAMBA_ENVIRONMENT}
echo "Mamba activated"
cd ${DIRECTORY}

export WANDB_CACHE_DIR=${TMPDIR}/wandb_cache
mkdir -p ${WANDB_CACHE_DIR}


# python src/main.py project_name=dmiise validation_period=50 trainer.max_epochs=600 dataloader.batch_size=128 dataloader.val_batch_size=128 diffusion=sample_diffusion dataset=acdc diffusion.repetitions=1 diffusion.repetitions_test=1 model.name=basic_unet metrics=metrics_handler_multi loss=x_0_pred_loss dataset.data_path=data/ACDC optimizer=optimizer_ema

# epsilon
# python src/main.py project_name=dmiise diffusion=base_diffusion diffusion.prediction_type=sample diffusion.noise_steps=100 diffusion.num_inference_steps=100 validation_period=30 trainer.max_epochs=600 dataloader.batch_size=64 dataloader.val_batch_size=256 dataset=mnist_labeled model.name=basic_unet metrics=metrics_handler_binary_guidance loss=mse_loss dataset.data_path=data/MNIST_modified optimizer=optimizer_ema optimizer.lr=0.0001 optimizer.weight_decay=0.00001 seed=3874 dataset.mask_transformer.train_switch=False

python src/main.py project_name=dmiise diffusion=base_diffusion diffusion.prediction_type=sample diffusion.noise_steps=100 diffusion.num_inference_steps=100 validation_period=30 trainer.max_epochs=300 dataloader.batch_size=16 dataloader.val_batch_size=64 dataset=bccd model.name=basic_unet model.emb_channels=256 metrics=metrics_handler_binary_guidance loss=x_0_pred_loss_binary dataset.data_path=data/BCCD optimizer=optimizer_ema optimizer.lr=0.001 optimizer.weight_decay=0.00000001 seed=3874 dataset.mask_transformer.train_switch=False


# python src/main.py project_name=dmiise diffusion=sample_diffusion diffusion.noise_steps=100 validation_period=50 trainer.max_epochs=600 dataloader.batch_size=64 dataloader.val_batch_size=256 dataset=acdc diffusion.repetitions=1 diffusion.repetitions_test=1 model.name=basic_unet metrics=metrics_handler_multi loss=x_0_pred_loss dataset.data_path=data/ACDC optimizer=optimizer_ema optimizer.lr=0.001 optimizer.weight_decay=0.00000001 seed=3874
# python main.py trainer.max_epochs=100 dataloader.batch_size=64 dataloader.val_batch_size=64 dataset/mask_transformer=acdc_multi


# python src/main.py project_name=dmiise diffusion=sample_diffusion diffusion.noise_steps=100 validation_period=20 trainer.max_epochs=300 dataloader.batch_size=64 dataloader.val_batch_size=128 dataset=bccd diffusion.repetitions=1 diffusion.repetitions_test=1 model.name=basic_unet metrics=metrics_handler_binary_guidance loss=x_0_pred_loss_binary dataset.data_path=data/BCCD optimizer=optimizer_ema
