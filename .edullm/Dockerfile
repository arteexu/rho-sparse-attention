ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

COPY pyproject.toml README.md ./
COPY rho_hf_pretrain.py rho_moe_experiment.py rho_moe_real_train.py rho_efficiency_sweep.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

CMD ["python", "-c", "import torch; print('rho image ready', torch.__version__)"]
