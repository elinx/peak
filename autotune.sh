#!/bin/bash

H_START=4
H_END=64
W_START=4
W_END=64
STEP=4

for ((h = $H_START; h <= $H_END; h += $STEP)); do
    for ((w = $H_START; w <= $H_END; w += $STEP)); do
        echo $h, $w
        sed -i "s/TILE_H \* \(.*\);/TILE_H \* $h;/g" manual_optimize_dgemm.cpp
        sed -i "s/TILE_W \* \(.*\);/TILE_W \* $w;/g" manual_optimize_dgemm.cpp
        cd build
        cost=`cmake --build . 2>&1 1>/dev/null && ./manual | grep time`
        echo $h, $w, $cost >> report.txt
        cd ..
    done
done