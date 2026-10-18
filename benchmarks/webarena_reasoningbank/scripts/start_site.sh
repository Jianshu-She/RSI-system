#!/bin/bash
# Create a FRESH shopping site from the image and start it.  Usage: scripts/start_site.sh <name> <offset>
#   offset k -> http://localhost:$((7770+k))  (mysql 3306+k, redis 6379+k, elasticsearch 9200+k)
# Every evaluation run should get its own fresh site: tasks can change site state (cart, orders, reviews).
set -e
source "$(dirname "$0")/../env.sh" > /dev/null
NAME=$1; OFF=$2; WEB=$((7770+OFF))
[ -z "$NAME" ] || [ -z "$OFF" ] && { echo "usage: $0 <name> <offset>"; exit 1; }
for port in $WEB $((3306+OFF)) $((6379+OFF)) $((9200+OFF)) $((9300+OFF)) $((9000+OFF)); do
  ss -ltn | grep -q ":$port " && { echo "port $port is in use (a site already runs on offset $OFF?). Stop it first: scripts/stop_site.sh <name> $OFF"; exit 1; }
done
if enroot list | grep -qx "$NAME"; then
  for p in $(pgrep -u $(id -u) -f "[s]upervisord -n"); do
    enroot exec $p grep -q "listen $WEB;" /etc/nginx/conf.d/default.conf 2>/dev/null && kill $p && sleep 15
  done
  enroot remove -f "$NAME"
fi
bash "$WA_HOME/scripts/make_site.sh" "$NAME" "$OFF" | tail -1
setsid nohup enroot start --rw "$NAME" >> "$WA_LOGS/site_$NAME.log" 2>&1 < /dev/null &
for i in $(seq 1 60); do        # wait for magento + mysql + elasticsearch
  sleep 10; PID=""
  for p in $(pgrep -u $(id -u) -f "[s]upervisord -n"); do
    enroot exec $p grep -q "listen $WEB;" /etc/nginx/conf.d/default.conf 2>/dev/null && PID=$p
  done
  [ -n "$PID" ] && curl -s -o /dev/null --max-time 5 "http://localhost:$((9200+OFF))" && \
    enroot exec $PID sh -c "mysqladmin -h127.0.0.1 -P$((3306+OFF)) -umagentouser -pMyPassword ping" > /dev/null 2>&1 && break
done
[ -z "$PID" ] && { echo "site $NAME did not start, see $WA_LOGS/site_$NAME.log"; exit 1; }
# Point Magento at this site's search engine + base URL, then require a real search hit. The image uses Magento's OpenSearch
# module: only catalog/search/opensearch_server_* takes effect (setting just elasticsearch7_server_port leaves every site
# querying localhost:9200, i.e. the offset-0 site, and search breaks when no offset-0 site runs). Retried because the
# config/flush can fail while Elasticsearch is still warming up; output goes to the site log (never fail silently).
SITELOG="$WA_LOGS/site_$NAME.log"; N=""
for try in 1 2 3 4 5 6; do
  enroot exec $PID sh -c "cd /var/www/magento2 && php bin/magento config:set catalog/search/opensearch_server_port $((9200+OFF)) \
    && php bin/magento config:set catalog/search/opensearch_server_hostname localhost \
    && php bin/magento config:set catalog/search/elasticsearch7_server_port $((9200+OFF)) \
    && php bin/magento setup:store-config:set --base-url=http://localhost:$WEB/ \
    && mysql -h127.0.0.1 -P$((3306+OFF)) -umagentouser -pMyPassword magentodb -e \"UPDATE core_config_data SET value='http://localhost:$WEB/' WHERE path='web/secure/base_url';\" \
    && php bin/magento cache:flush" >> "$SITELOG" 2>&1 || echo "[start_site] config attempt $try failed" >> "$SITELOG"
  N=$(curl -s --max-time 120 "http://localhost:$WEB/catalogsearch/result/?q=shoe+storage" | grep -o 'toolbar-number">[0-9,]*' | tail -1 | grep -o '[0-9,]*$' || true)
  [ "$N" = "19120" ] && break
  echo "[start_site] search check attempt $try: '$N' hits (expected 19120), retrying" >> "$SITELOG"; sleep 20
done
if [ "$N" = "19120" ]; then echo "OK site $NAME at http://localhost:$WEB (search check 19120 hits)"
else echo "FAILED site $NAME: search check gave '$N' hits (expected 19120), see $WA_LOGS/site_$NAME.log"; exit 1; fi
