#!/bin/bash

bash initialize-caddy-certificate.sh
bash initialize-ssh-host-keys.sh
bash initialize-initial-compile-cache.sh
service ssh start
exec /bin/bash
