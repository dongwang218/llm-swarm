#!/bin/bash
#SBATCH --job-name=llm-swarm
#SBATCH --ntasks=8              # One task per GPU
#SBATCH --gpus-per-task=1       # Each task gets one GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=11G
#SBATCH -o slurm/logs/%x_%j.out
#SBATCH --account=data
#SBATCH --qos=data_high

# For HF cluster internal users: Check if /fsx directory exists
if [ -d "/fsx/.cache" ]; then
    export volume="/fsx/.cache"
else
    export volume=".cache"
fi
export model={{model}}
export revision={{revision}}
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    # try reading from file
    export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
fi
# unset cache dirs to avoid pyxis having host env var somehow get into the container
unset HF_HUB_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0

srun bash -c '

    function unused_port() {
        N=${1:-1}
        comm -23 \
            <(seq 1025 65535 | sort) \
            <(ss -Htan |
                awk "{print \$4}" |
                cut -d":" -f2 |
                sort -u) |
            shuf |
            head -n "$N"
    }

    export PORT=$(unused_port)
    echo "Starting TGI container port $PORT on host $(hostname)"
    echo "http://$(hostname -I | awk '\''{print $1}'\''):$PORT" >> {{slurm_hosts_path}}
    python3 -m vllm.entrypoints.openai.api_server \
        --model $model \
        --port $PORT \
        --host "0.0.0.0" \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.8 \
        --enable-prefix-caching
'
echo "End of job"