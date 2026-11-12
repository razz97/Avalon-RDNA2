#!/usr/bin/env bash
# the A/B sequence used for PERFORMANCE.md (serial; each run = fresh server)
cd "$(dirname "$0")/.."
for spec in "B 65536" "A 65536" "B 126972" "A 126972" "B 40960" "A 40960" "B 40960" "A 40960"; do
  set -- $spec
  scripts/run-ab-bench.sh "$1" "$2" 8192 4096 > "results/logs/ab-$1-$2-$(date +%H%M%S).log" 2>&1 < /dev/null || echo "FAILED $spec" >> results/logs/ab-chain.fail
done
echo done > results/logs/ab-chain.done
