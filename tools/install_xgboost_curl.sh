#!/usr/bin/env bash
set -euo pipefail

WHEEL=/tmp/xgboost-1.7.6-py3-none-manylinux2014_x86_64.whl
URL=https://pypi.tuna.tsinghua.edu.cn/packages/8c/3a/c9c5d4d5c49b132ef15ac7b5ccf56ef1c82efe36cd19414771762e97c00e/xgboost-1.7.6-py3-none-manylinux2014_x86_64.whl

rm -f "$WHEEL"
curl -L --retry 10 --connect-timeout 20 --output "$WHEEL" "$URL"
/home/busanbusi/.virtualenvs/experiment/bin/python -m pip install "$WHEEL"
/home/busanbusi/.virtualenvs/experiment/bin/python -c "import xgboost; print(xgboost.__version__)"
