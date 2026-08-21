# Hailo Dataflow Compiler 5.3.0, on the platform its wheel demands.
#
# The wheel is tagged `py3-none-linux_x86_64`: no Python-version pin, but Linux on
# x86-64 and nothing else. macOS on arm64 cannot run it, which is why this exists.
#
# Python 3.10 is not a guess. The wheel pins tensorflow==2.19.1, numpy==1.26.4 and
# onnxruntime==1.18.0, and 3.10 is inside every one of those ranges. Ubuntu 22.04
# ships 3.10 as its system Python, so the base image and the wheel agree already.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      build-essential graphviz libgraphviz-dev libgl1 libglib2.0-0 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/dfc
ENV PATH=/opt/dfc/bin:$PATH PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir --upgrade "pip<25" setuptools wheel

WORKDIR /work

# CPU torch FIRST, so the DFC's `torch==2.9.1` pin is already satisfied when its wheel
# is installed. PEP 440 matches `2.9.1+cpu` against `==2.9.1`, so the pin holds and the
# CUDA stack never arrives.
#
# This is not tidiness. The default resolution pulls cuBLAS, cuDNN, NCCL, cuSPARSE,
# cuSOLVER, triton and six more nvidia wheels, which took the image to 5.1 GB
# compressed and past Fly's 8 GB uncompressed unpack limit -- the machine failed to
# start with "Not enough space to unpack image". Parsing and emulation are CPU work and
# there is no GPU on this machine, so every one of those bytes was unreachable.
RUN pip install --no-cache-dir --no-compile \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.9.1 torchvision==0.24.1

COPY hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir --no-compile \
      /tmp/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl \
 && rm /tmp/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl \
 && pip cache purge || true

# The DFC drags in the whole Jupyter stack as a dependency and nothing here runs a
# notebook. Removing it and the bundled test trees is the difference between fitting
# the unpack limit and not.
RUN python - <<'EOF'
import shutil, pathlib, sys
sp = pathlib.Path("/opt/dfc/lib/python3.10/site-packages")
freed = 0
for name in ("notebook", "notebook_shim", "jupyterlab", "jupyter_server",
             "jupyterlab_server", "nbconvert", "nbclient", "ipywidgets",
             "widgetsnbextension", "jupyterlab_widgets", "jupyter_console"):
    d = sp / name
    if d.exists():
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
print(f"removed {freed/1e6:.0f} MB of notebook packages", file=sys.stderr)
EOF

# The import the gate needs must survive the pruning above. A build that ships a
# broken compiler and finds out on the machine has moved the failure, not removed it.
RUN python -c "from hailo_sdk_client import ClientRunner; print('hailo_sdk_client imports OK')"

# The graphs the macOS gate exported, and the gate that checks the compiler agrees.
COPY models/ /work/models/
COPY scripts/ /work/scripts/

# The machine stays up so `fly ssh console` can drive the compiler interactively.
CMD ["sleep", "infinity"]
