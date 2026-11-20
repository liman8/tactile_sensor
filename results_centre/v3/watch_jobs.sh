#!/bin/bash
# Emits new interesting lines from the v3 logs (state kept across re-arms), stalls, low memory, OOM kills.
D="$(dirname "$(readlink -f "$0")")"
ST=$D/.watch_state
mkdir -p $ST
PAT='best epoch|epoch   1:|epoch  (40|80):|epoch 120:|net:|success|Traceback|Error|BrokenProcess|Killed|CUDA|nan|wrote|missing|size .* seed|skip|DONE'
since=$(cat $ST/since 2>/dev/null || date '+%Y-%m-%d %H:%M:%S'); echo "$since" > $ST/since
while true; do
  for f in $D/train_*_s*.log $D/compare*.log $D/search_*.log $D/curve*.log $D/seeds*.log $D/runtime*.log $D/*.done; do
    [ -f "$f" ] || continue
    b=$(basename $f); n=$(wc -l < "$f"); o=$(cat "$ST/$b" 2>/dev/null || echo 0)
    if [ "$n" -gt "$o" ]; then tail -n +$((o + 1)) "$f" | head -n $((n - o)) | grep -E "$PAT" | grep -v Warn | sed "s|^|[$b] |" | cut -c1-260; echo $n > "$ST/$b"; fi
  done
  # stall: newest training log not modified for 15 min while a training process runs
  t=$(ls -t $D/train_*_s*.log 2>/dev/null | head -1)
  if [ -n "$t" ] && pgrep -f "[c]entre_v3.py train" >/dev/null; then
    age=$(( $(date +%s) - $(stat -c %Y "$t") ))
    if [ $age -gt 2700 ] && [ ! -f $ST/stall_$(basename $t) ]; then echo "STALL: $(basename $t) not updated for $((age/60)) min"; touch $ST/stall_$(basename $t); fi
  fi
  av=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$av" -lt 5 ] && [ ! -f $ST/lowmem ]; then echo "LOW MEMORY: ${av} GB available"; touch $ST/lowmem; fi
  [ "$av" -ge 8 ] && rm -f $ST/lowmem
  k=$(journalctl -k --since "$since" --no-pager 2>/dev/null | grep -E "Killed process|D-Notifier|Xid" | tail -3)
  [ -n "$k" ] && { echo "KERNEL: $k" | cut -c1-300; since=$(date '+%Y-%m-%d %H:%M:%S'); echo "$since" > $ST/since; }
  sleep 30
done
