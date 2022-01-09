#!/bin/bash

#python driver.py -c $1 -s &&
#rm -rf bag_analysis &&
#mkdir bag_analysis &&
#mkdir bag_analysis/bags &&
#mv *.bag ./bag_analysis/bags/ &&
python analyze_bag.py --input ./bag_analysis_OK_GPU1/bags/ --output ./bag_analysis_OK_GPU1/
