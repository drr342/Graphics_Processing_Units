#!/bin/bash

echo "1000 1000"
nvprof ./heatdist 1000 1000 1
echo ""
echo "2000 1000"
nvprof ./heatdist 2000 1000 1
echo ""
echo "4000 1000"
nvprof ./heatdist 4000 1000 1
echo ""
echo "8000 1000"
nvprof ./heatdist 8000 1000 1
echo ""
echo "16000 1000"
nvprof ./heatdist 16000 1000 1
echo ""
echo "1000 2000"
nvprof ./heatdist 1000 2000 1
echo ""
echo "1000 4000"
nvprof ./heatdist 1000 4000 1
echo ""
echo "1000 8000"
nvprof ./heatdist 1000 8000 1
echo ""
echo "1000 16000"
nvprof ./heatdist 1000 16000 1
echo ""
echo "1000 32000"
nvprof ./heatdist 1000 32000 1
echo ""





