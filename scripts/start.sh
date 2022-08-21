#!/bin/sh

JUPYTER_DIR=/home/vscode/.jupyter
S3_MOUNT=/mnt/s3

mkdir -p $JUPYTER_DIR
mkdir -p $S3_MOUNT

git config --global credential.helper "store --file /home/vscode/.creds/.git-credentials"
git config --global --add safe.directory '*'

goofys -f --cheap \
    --stat-cache-ttl 30m \
    ${DEEPLAB_BUCKETNAME}:.jupyter/ $JUPYTER_DIR &
goofys -f --cheap \
    --stat-cache-ttl 30m \
    ${DEEPLAB_BUCKETNAME}:dataset/ $S3_MOUNT &
(
    echo launching jupyter lab...
    # wait for goofys mounting
    sleep 5

    # launch jupyter lab
    python3 -m jupyter lab \
        --ContentsManager.allow_hidden=True \
        --port 5515 --ip=0.0.0.0 --allow-root \
        --notebook-dir=/workspace
)
