#!/bin/bash

echo "Scaling Sample Problem on CPU: Training Data"
./svm-scale -s param.cpu sampleData.train
echo ""

echo "Training Sample Problem on CPU"
./svm-train sampleData.train.scale.cpu model
echo ""

echo "Scaling Sample Problem on CPU: Test Data"
./svm-scale -r param.cpu sampleData.test
echo ""

echo "Predicting Sample Problem on CPU"
./svm-predict sampleData.test.scale.cpu model.cpu prediction
echo ""

echo "============================================"
echo ""

echo "Scaling Sample Problem on GPU: Training Data"
./svm-scale-gpu -s param.gpu sampleData.train
echo ""

echo "Training Sample Problem on GPU"
./svm-train-gpu sampleData.train.scale.gpu model
echo ""

echo "Scaling Sample Problem on GPU: Test Data"
./svm-scale-gpu -r param.gpu sampleData.test
echo ""

echo "Predicting Sample Problem on GPU"
./svm-predict-gpu sampleData.test.scale.gpu model.gpu prediction
echo ""
