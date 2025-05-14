#!/bin/bash
INPUT="input.png"
THREADS=(1 2 4 8 16 32 48)
RES=("389x518" "778x1036" "1167x1554" "1556x2072" "1945x2590" "2334x3108" "2723x3626")

echo "resolution,threads,stage1,stage2,stage3,stage4,stage5,stage6,stage7,total" > results.csv
for r in "${RES[@]}"; do
  for t in "${THREADS[@]}"; do
    ./parallelcode3 "$INPUT" "$t" "$r" >> results.csv
  done
done
