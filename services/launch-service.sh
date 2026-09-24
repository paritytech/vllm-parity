#!/bin/bash

ARCHIVE_DIR=/workspace/archive
SERVICE="$(basename -- "${BASH_SOURCE[1]:?must be sourced from another script}" .sh)"
SERVICE_LOG="/workspace/$SERVICE.log"

screen -wipe >/dev/null 2>&1 || true

if screen -list 2>/dev/null | grep -q "\.${SERVICE}[[:space:]]"; then
    echo "screen session '$SERVICE' already exists; will not launch another one" >&2
    exit 1
fi

rotate_logs() {
    local stamp file name stem extension destination attempt

    stamp="$(date -u +%Y%m%dT%H%M%SZ)"

    for file in "$@"; do
        if [ -f "$file" ]; then
            name="${file##*/}"
            stem="${name%.*}"
            extension="${name##*.}"
            destination="$ARCHIVE_DIR/$stem-$stamp.$extension"
            attempt=0

            while [ -e "$destination" ]; do
                attempt=$((attempt + 1))
                destination="$ARCHIVE_DIR/$stem-$stamp-$attempt.$extension"
            done

            mkdir -p "$ARCHIVE_DIR"
            mv -- "$file" "$destination"
            echo "Archive: $file -> $destination"
        fi

        touch "$file"
    done
}

launch_service() {
    screen -dmS "$SERVICE" bash with-logging.sh "$SERVICE_LOG" "$@"
}
