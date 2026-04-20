#!/usr/bin/env bash
set -euo pipefail

HOT=/mnt/nvme_raid0/experiment_data
COLD=/mnt/hdd_data
LOG=/home/busanbusi/experiment/hot_data_sync.log

mkdir -p \
  "$HOT/tick/stock_tick_month" \
  "$HOT/l2/order" \
  "$HOT/min/1m_hfq" \
  "$HOT/day" \
  "$HOT/adj_factor" \
  "$HOT/logs"

exec >> "$LOG" 2>&1

echo "==== sync start $(date -Is) ===="
echo "hot=$HOT cold=$COLD"
df -hT "$HOT" "$COLD" || true

copy_dir() {
  local src="$1"
  local dst_parent="$2"

  if [ ! -d "$src" ]; then
    echo "MISSING $src $(date -Is)"
    return 0
  fi

  echo "COPY $src -> $dst_parent $(date -Is)"
  mkdir -p "$dst_parent"
  if command -v rsync >/dev/null 2>&1; then
    ionice -c2 -n7 nice -n 10 rsync -a --ignore-existing --stats "$src" "$dst_parent/"
  else
    ionice -c2 -n7 nice -n 10 cp -an "$src" "$dst_parent/"
  fi
  echo "DONE $src $(date -Is)"
  du -sh "$dst_parent" || true
}

for y in 2021 2022 2023; do
  copy_dir "$COLD/A股_分时数据/A股_分时数据_沪深/1分钟_前复权_按年汇总/${y}_1min" "$HOT/min/1m_hfq"
done

for y in 2021 2022 2023; do
  copy_dir "$COLD/tick/stock_tick_month/$y" "$HOT/tick/stock_tick_month"
done

for y in 2021 2022 2023; do
  copy_dir "$COLD/l2/order/$y" "$HOT/l2/order"
done

echo "==== sync done $(date -Is) ===="
du -sh "$HOT" "$HOT"/* || true
