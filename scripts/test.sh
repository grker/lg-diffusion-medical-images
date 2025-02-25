#!/usr/bin/env bash
#SBATCH -p lowprio
#SBATCH --mail-type=NONE
#SBATCH --job-name=ModelTraining
#SBATCH --time=00:30:00
#SBATCH --output=/data/%u/master_thesis/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/data/%u/master_thesis/jobs/%j.err # where to store error messages
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=0


source /etc/profile.d/lmod.sh
source /etc/profile.d/spack.sh
source /etc/profile.d/slurm.sh

# Exit on errors
set -o errexit

module load gpu python/3.11 cuda/12.4.1 mamba


# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-23.11.0-0-334ztq7i4mzu762ew2x3kbbrrorhe6eg/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-23.11.0-0-334ztq7i4mzu762ew2x3kbbrrorhe6eg/etc/profile.d/conda.sh" ]; then
        . "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-23.11.0-0-334ztq7i4mzu762ew2x3kbbrrorhe6eg/etc/profile.d/conda.sh"
    else
        export PATH="/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-23.11.0-0-334ztq7i4mzu762ew2x3kbbrrorhe6eg/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-23.11.0-0-334ztq7i4mzu762ew2x3kbbrrorhe6eg/etc/profile.d/mamba.sh" ]; then
    . "/apps/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0/mamba-23.11.0-0-334ztq7i4mzu762ew2x3kbbrrorhe6eg/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi

trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' INT TERM EXIT

export TMPDIR

USERNAME=$USER
PROJECT_NAME=master_thesis
DIRECTORY=/data/${USERNAME}/${PROJECT_NAME}/src
MAMBA_ENVIRONMENT=master_thesis

mamba activate ${MAMBA_ENVIRONMENT}
echo "Mamba activated"
cd ${DIRECTORY}


export TQDM_DISABLE="1"
export HYDRA_FULL_ERROR=1

export WANDB_CACHE_DIR=${TMPDIR}/wandb_cache
mkdir -p ${WANDB_CACHE_DIR}

python test.py

rm -rf $TMPDIR

exit 0
