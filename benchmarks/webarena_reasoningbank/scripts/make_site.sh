#!/bin/bash
# Create a fresh, independent WebArena shopping instance from shopping.sqsh (enroot, non-root).
# Usage: make_shopping_instance.sh <name> <port_offset>
#   offset 0 -> site on localhost:7770 (mysql 3306, redis 6379, es 9200/9300, php-fpm 9000)
#   offset k -> every port + k, site on localhost:$((7770+k))
set -e
NAME=$1; OFF=$2
source "$(dirname "$0")/../env.sh" > /dev/null
SQSH=$WA_SITE_IMAGE
R=$ENROOT_DATA_PATH/$NAME
[ -d "$R" ] && { echo "$R exists; remove it first (enroot remove -f $NAME)"; exit 1; }
enroot create -n "$NAME" "$SQSH"

WEB=$((7770+OFF)); MY=$((3306+OFF)); RD=$((6379+OFF)); ESH=$((9200+OFF)); EST=$((9300+OFF)); FPM=$((9000+OFF)); CTRL=$((8877+OFF))

# --- run unprivileged ---
sed -i 's/^user  nginx;/# user nginx;/; s|^pid .*|pid /tmp/nginx.pid;|' $R/etc/nginx/nginx.conf
sed -i "s/listen 80;/listen $WEB;/; s/listen \[::\]:80 ipv6only=on default_server;/# ipv6 disabled/; s/server  127.0.0.1:9000;/server  127.0.0.1:$FPM;/" $R/etc/nginx/conf.d/default.conf
sed -i "s/listen 80 default_server;/listen $((7799-OFF)) default_server;/; s/listen \[::\]:80 default_server;/# ipv6 disabled/" $R/etc/nginx/http.d/default.conf
sed -i "s|^command=su elastico -c \"ES_JAVA_HOME=/usr elasticsearch\"|command=sh -c \"ES_JAVA_HOME=/usr ES_JAVA_OPTS='-Xms4g -Xmx4g' elasticsearch\"|" $R/etc/supervisor.d/elasticsearch.ini
sed -i 's/^command=mysqld --user=mysql/command=mysqld/' $R/etc/supervisor.d/mysql.ini
sed -i "s/^command=redis-server$/command=redis-server --bind 127.0.0.1 --protected-mode yes --port $RD/" $R/etc/supervisor.d/redis.ini
mv $R/etc/supervisor.d/mailcatcher.ini $R/etc/supervisor.d/mailcatcher.off
mv $R/etc/supervisor.d/cron.ini $R/etc/supervisor.d/cron.off
sed -i 's|^file=/run/supervisord.sock|file=/tmp/supervisord.sock|; s|^serverurl=unix:///run/supervisord.sock|serverurl=unix:///tmp/supervisord.sock|' $R/etc/supervisord.conf
sed -i 's/^user = www-data/;user = www-data/; s/^group = www-data/;group = www-data/' $R/usr/local/etc/php-fpm.d/www.conf
sed -i "s/^listen = 9000/listen = 127.0.0.1:$FPM/" $R/usr/local/etc/php-fpm.d/zz-docker.conf
sed -i "s/^listen = 127.0.0.1:9000/listen = 127.0.0.1:$FPM/" $R/usr/local/etc/php-fpm.d/www.conf
sed -i "s/^#bind-address=0.0.0.0/bind-address=127.0.0.1\nport=$MY/" $R/etc/my.cnf.d/mariadb-server.cnf
printf '[client]\nport=%s\n' $MY >> $R/etc/my.cnf
cat >> $R/usr/share/java/elasticsearch/config/elasticsearch.yml <<EOF
cluster.name: wa-$NAME
discovery.type: single-node
network.host: 127.0.0.1
http.port: $ESH
transport.port: $EST
EOF
sed -i "s/^WA_ENV_CTRL_PORT=.*/WA_ENV_CTRL_PORT=$CTRL/; s/^WA_ENV_CTRL_ENABLE=.*/WA_ENV_CTRL_ENABLE=false/" $R/etc/environment
# magento: db / redis ports
ENVPHP=$R/var/www/magento2/app/etc/env.php
sed -i "s/'host' => '127.0.0.1',/'host' => '127.0.0.1:$MY',/; s/'port' => '6379',/'port' => '$RD',/g" $ENVPHP
printf 'mkdir -p /run/mysqld /run/nginx /run/redis /var/tmp/nginx/client_body\nexec supervisord -n -c /etc/supervisord.conf\n' > $R/etc/rc.local
echo "created $NAME: site http://localhost:$WEB (mysql $MY, redis $RD, es $ESH)"
