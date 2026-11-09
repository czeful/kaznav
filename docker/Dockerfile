# kznav-sim: the stand's simulator environment (conda habitat_env: Habitat-Sim 0.2.4 headless + bullet, torch 2.8)
# and the model-server venv (qolda-env: the Qolda, Qwen3-VL and Gemma 3 servers that tools/run_matrix.py starts),
# built from the lock files in env/ (docker/lock_envs.sh). Code, datasets and model weights are not in the image:
# docker/compose.yaml mounts them. Build: bash docker/kz.sh --build
FROM ubuntu:22.04

# MESA_VERSION pins Mesa (WSL2 render path, EGL -> D3D12) to the build the dev results were rendered with;
# "" takes the current jammy-updates build.
ARG MESA_VERSION=23.2.1-1ubuntu3.1~22.04.4
ARG MICROMAMBA_VERSION=2.0.5-0
ENV DEBIAN_FRONTEND=noninteractive

# What the environments link against (ldd of habitat_sim and cv2), Mesa, curl for the server scripts, git and a
# compiler for the pip packages that ship only sources.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl bzip2 git build-essential \
        libegl1 libopengl0 libgl1 libglx0 libglvnd0 libgomp1 libglib2.0-0 libx11-6 \
        libegl-mesa0${MESA_VERSION:+=$MESA_VERSION} libgl1-mesa-dri${MESA_VERSION:+=$MESA_VERSION} \
        libglapi-mesa${MESA_VERSION:+=$MESA_VERSION} libgbm1${MESA_VERSION:+=$MESA_VERSION} \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /usr/local/bin/micromamba \
        "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64" \
    && chmod +x /usr/local/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda

# habitat_env: the conda packages by URL and md5, then what pip installed on top of them.
COPY env/habitat_env.conda.txt env/habitat_env.pip.txt docker/pip_lock_install.sh /tmp/lock/
RUN micromamba create -y -p /opt/conda/envs/habitat_env -f /tmp/lock/habitat_env.conda.txt \
    && bash /tmp/lock/pip_lock_install.sh /opt/conda/envs/habitat_env/bin/python /tmp/lock/habitat_env.pip.txt \
    && micromamba clean -a -y

# qolda-env: a venv on the Python build it was made with (3.13.11 from pkgs/main), every package pinned.
COPY env/qolda-env.pip.txt /tmp/lock/
RUN micromamba create -y -p /opt/py313 -c https://repo.anaconda.com/pkgs/main "python=3.13.11=hcf712cf_100_cp313" \
    && /opt/py313/bin/python -m venv /opt/qolda-env \
    && bash /tmp/lock/pip_lock_install.sh /opt/qolda-env/bin/python /tmp/lock/qolda-env.pip.txt \
    && micromamba clean -a -y && rm -rf /tmp/lock

ENV PATH=/opt/conda/envs/habitat_env/bin:$PATH \
    CONDA_PREFIX=/opt/conda/envs/habitat_env \
    KZNAV_QOLDA_VENV=/opt/qolda-env KZNAV_QWEN_VENV=/opt/qolda-env KZNAV_GEMMA_VENV=/opt/qolda-env \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    HF_HUB_DISABLE_XET=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 USER=kznav
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/kznav-entrypoint
ENTRYPOINT ["/usr/local/bin/kznav-entrypoint"]
CMD ["bash"]
