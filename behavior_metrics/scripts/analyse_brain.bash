#!/bin/bash

python driver.py -c ./configs/torch-multiple.yml -s &&
rm -rf bag_analysis &&
mkdir bag_analysis &&
mkdir bag_analysis/bags && 
mv *.bag ./bag_analysis/bags/ &&
python analyze_bag.py --input ./bag_analysis/bags/ --output ./bag_analysis/