#!/bin/bash
# Stop and delete a site.  Usage: scripts/stop_site.sh <name> <offset>
source "$(dirname "$0")/../env.sh" > /dev/null
NAME=$1; WEB=$((7770+$2))
for p in $(pgrep -u $(id -u) -f "[s]upervisord -n"); do
  enroot exec $p grep -q "listen $WEB;" /etc/nginx/conf.d/default.conf 2>/dev/null && kill $p
done
sleep 15; enroot remove -f "$NAME" && echo "removed $NAME"
