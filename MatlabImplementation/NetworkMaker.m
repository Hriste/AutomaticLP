clear all
close all

% this tutorial was suprisingly difficult to find
% https://www.mathworks.com/help/vision/ug/create-yolo-v2-object-detection-network.html
% Let's attempt to create a network

% Load a pretrained network
net = mobilenetv2();

% Convert network into a layer graph object 
% in order to manipulate layers
lgraph = layerGraph(net);

% Input Size - this is specific to my network !
% TODO: switch back to 2:1, I thougth this might fix a matlab error - did
% not
imageInputSize = [512, 256, 3];

% Create a new input image layer
% Set the name to the original layer name
imgLayer = imageInputLayer(imageInputSize, "Name", "input_1");

% Replace old image input layer
lgraph = replaceLayer(lgraph, "input_1", imgLayer);

% Good feature extraction layer is where the output 
% feature width and height is between 8 to 16 times
% smaller than the input image. 
% Downsampling is a trade-off between spatial resilution
% and quality of output features.

% so this layer is [19,19] times 16 this is about 300
% because we have a 2:1 ratio this is kinda inbetween
% and since this is used in the tutorial I'm sticking
% with it for now
featureExtractionLayer = "block_12_add";

% load a network modified by removing layers after feature 
% extraction - take the layers off with the deep network designer
modified = load('modifiedMobileNetv2.mat');
lgraph = modified.lgraph_1;

% Create YOLOv2 Detection Sub-Network

% Set the convolution layer filter size to [3 3]
% This size is common in CNN architectures
filterSize = [3,3];

% Set the number of filters in the convolution layers to match the number
% of channels in the feature extrraction layer output
numFilters = 74%148; 

% Create the detection subnetwork
% the convolution layer uses the same "padding" to preserve the input size
detectionLayers = [
  % group 1
  convolution2dLayer(filterSize, numFilters, "Name", "yolov2Conv1", ...
  "Padding", "same", "WeightsInitializer", @(sz)rand(sz)*0.01)
  batchNormalizationLayer("Name", "yolov2Batch1");
  reluLayer("Name", "yolov2Relu1");
  
  % group 2
  convolution2dLayer(filterSize, numFilters, "Name", "yolo2Conv2",...
  "Padding", "same", "WeightsInitializer", @(sz)rand(sz)*0.01)
  batchNormalizationLayer("Name", "yolov2Batch2");
  reluLayer("Name", "yolov2Relu2");
];

% define the number of classes to detect
numClasses = 32%32;

% define the anchor boxes
anchorBoxes = load('formattedData.mat').anchorBoxes;

numAnchors = size(anchorBoxes,1);

% There are five predictions per anchor box: 
%  * Predict the x, y, width, and height offset
%    for each anchor.
%  * Predict the intersection-over-union with ground
%    truth boxes.
numPredictionsPerAnchor = 5;

% Number of filters in last convolution layer.
outputSize = numAnchors*(numClasses+numPredictionsPerAnchor);

% Final layers in detection sub-network.
finalLayers = [
    convolution2dLayer(1,outputSize,"Name","yolov2ClassConv",...
    "WeightsInitializer", @(sz)randn(sz)*0.01)
    yolov2TransformLayer(numAnchors,"Name","yolov2Transform")
    yolov2OutputLayer(anchorBoxes,"Name","yolov2OutputLayer")
    ];

% Add the last layers to network.
detectionLayers = [
    detectionLayers
    finalLayers
    ];
% Add the detection subnetwork to the feature extraction network.
lgraph = addLayers(lgraph,detectionLayers);

% Connect the detection subnetwork to the feature extraction layer.
lgraph = connectLayers(lgraph,featureExtractionLayer,"yolov2Conv1");

% save off custom network
save('customNetwork.mat','lgraph');