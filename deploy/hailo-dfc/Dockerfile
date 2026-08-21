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
ENV PATH=/opt/dfc/bin:$PATH
RUN pip install --no-cache-dir --upgrade "pip<25" setuptools wheel

WORKDIR /work
COPY hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl \
 && rm /tmp/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl

# The graphs the macOS gate exported, and the gate that checks the compiler agrees.
COPY models/ /work/models/
COPY scripts/ /work/scripts/

# The machine stays up so `fly ssh console` can drive the compiler interactively.
CMD ["sleep", "infinity"]
