#!/bin/bash
cd "$(dirname "$0")"

BASE_DIR=/data/dataset

TIMEFORMAT=%R

find "$BASE_DIR" -type f | grep -v "nlang.yaml" | while read -r file; do
    RETURN_VALUE=$( { time /app/build/libtglang-tester/libtglang-tester "$file" ; } 2>&1 )
    PREDICTION=$(echo "$RETURN_VALUE" | head -n -1)
    EXEC_TIME=$(echo "$RETURN_VALUE" | tail -n 1)
    FILE_LENGTH=$(wc -c < "$file")
    echo "$EXEC_TIME,$PREDICTION,$file,$FILE_LENGTH"
done
