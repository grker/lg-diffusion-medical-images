#!/usr/bin/env bash
#SBATCH -p lowprio
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, AL
#SBATCH --output=/data/%u/master_thesis/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/data/%u/master_thesis/jobs/%j.err # where to store error messages
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

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
echo "Mamba activated using the environment ${MAMBA_ENVIRONMENT}"
cd ${DIRECTORY}

export WANDB_CACHE_DIR=${TMPDIR}/wandb_cache
mkdir -p ${WANDB_CACHE_DIR}

# python src/test_ensemble.py run_id=29bjivsr 'repetitions=[1]'
python src/loss_guidance.py run_id=29bjivsr loss_guidance.starting_step=5
