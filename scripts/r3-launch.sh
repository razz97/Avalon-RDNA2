#!/usr/bin/env bash
# Run a script as a transient user service OUTSIDE the Hermes worker scope (that scope has MemoryMax=4G, which makes the 90 GB mmapped model thrash the page cache).
# Usage: r3-launch.sh NAME CMD...   (log: results-r3/NAME.log; wait for it with: systemctl --user is-active NAME)
set -eu; n=$1; shift; cd "$(dirname "$0")/.."
systemd-run --user --unit="$n" --collect --working-directory="$PWD" -p StandardOutput=append:"$PWD/results-r3/$n.log" -p StandardError=inherit "$@"
