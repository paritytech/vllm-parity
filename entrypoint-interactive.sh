#!/bin/bash

bash initialize-caddy-certificate.sh
bash services/sshd.sh
bash restore-cache.sh
exec /bin/bash
