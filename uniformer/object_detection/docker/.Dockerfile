FROM nvcr.io/nvidia/tensorrt:22.04-py3
RUN pip3 install torch==1.10.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    && pip3 install timm
    && pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN apt-get update && apt-get install -y libgl1-mesa-glx