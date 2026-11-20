#!/bin/bash
# Training watchdog: exits (so the caller is notified) when the training log has not changed for > LIMIT seconds
# while the training process still runs, or when the process is gone. Usage: watchdog.sh LOGFILE PROC_PATTERN [LIMIT]
LOG=$1; PAT=$2; LIMIT=${3:-600}
while true; do
  # alive = a process other than a shell / grep / awk / ps carries PAT in its command line (this watchdog and
  # whoever waits on it carry PAT too, so pgrep -f alone would always find something)
  if ! ps -eo comm=,args= | awk -v p="$PAT" '$1 !~ /^(bash|sh|grep|awk|ps|pgrep)$/ && index($0, p) { f = 1 } END { exit !f }'; then echo "$(date '+%F %T') process gone: $PAT"; exit 0; fi
  age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || date +%s) ))
  if [ $age -gt $LIMIT ]; then
    echo "$(date '+%F %T') STALE: $LOG not written for $age s"
    journalctl -k --since "-30min" --no-pager 2>&1 | grep -iE "nvrm|nvidia|acpi|xid|gpu" | tail -20
    nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,power.limit --format=csv,noheader 2>&1
    exit 0
  fi
  sleep 60
done
