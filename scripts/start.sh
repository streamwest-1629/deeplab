#!/bin/sh

JUPYTER_DIR=/home/vscode/.jupyter
S3_MOUNT=/mnt/s3

rm -r $JUPYTER_DIR
rm -r $S3_MOUNT

mkdir -p $JUPYTER_DIR
mkdir -p $S3_MOUNT

git config --global credential.helper "store --file /home/vscode/.git-credentials"
git config --global --add safe.directory '*'

/workspace/.devcontainer-venv/bin/python -m pip install -r "/workspace/requirements/pytorch.requirements.lock" \
    --extra-index-url https://download.pytorch.org/whl/cu113

goofys -f --cheap \
    --stat-cache-ttl 30m \
    ${DEEPLAB_BUCKETNAME}:.jupyter/ $JUPYTER_DIR &
goofys -f --cheap \
    --stat-cache-ttl 30m \
    ${DEEPLAB_BUCKETNAME}:dataset/ $S3_MOUNT &
(
    # for vscode
    cd /workspace
    npm ci &&
    npm start
)
# (
#     echo launching jupyter lab...
#     # wait for goofys mounting
#     sleep 5

#     # launch jupyter lab
#     python3 -m jupyter lab \
#         --ContentsManager.allow_hidden=True \
#         --Completer.use_jedi=False \
#         --port 5515 --ip=0.0.0.0 --allow-root \
#         --notebook-dir=/workspace
# )
