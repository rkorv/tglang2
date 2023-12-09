#!/bin/bash
cd "$(dirname "$0")"/..

REPORT_DIR="`pwd`/../report"
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
REPORT_DIR_WITH_DATE="$REPORT_DIR/$CURRENT_DATETIME"

DATASETS_ROOT=`pwd`/../../datasets
DATASETS_LIST="$DATASETS_ROOT/CPP_TEST $DATASETS_ROOT/llama/tasks"

# Build environment
docker build -f ./test.Dockerfile -t tglang_test .

# Build libtglang-tester
docker run --rm -it -v `pwd`/../libtglang.so:/build/libtglang/libtglang.so -v `pwd`:/app tglang_test \
    bash -c "cd /app && mkdir -p build/libtglang-tester && cd build/libtglang-tester && cmake ../../libtglang-tester && make"

mkdir -p $REPORT_DIR_WITH_DATE
touch $REPORT_DIR_WITH_DATE/test_results.csv
touch $REPORT_DIR_WITH_DATE/test_results_analysis.txt

# Run testing script
for dataset in $DATASETS_LIST; do
    echo "Testing $dataset"
    docker run --rm -it -v `pwd`/../libtglang.so:/build/libtglang/libtglang.so \
        -v $dataset:/data/dataset -v $REPORT_DIR_WITH_DATE/:/data/report -v `pwd`:/app --cpus 8 tglang_test \
        bash -c "/app/scripts/tester.sh | tee -a /data/report/test_results.csv"
done

# Run analysis script
python ./scripts/analyse.py $REPORT_DIR_WITH_DATE/test_results.csv | tee $REPORT_DIR_WITH_DATE/test_results_analysis.txt
