close all
clear all

% Run the python generator tool & the Conversion.m script to get the
% dataset
% TODO: make conversion a function call
% TODO: possibly call python script from matlab file
trainingData = load('formattedData.mat').dsnew;
%trainingData = load('formattedData.mat').sc;
%trainingData = load('formattedData.mat').tc;

% temp = load('imageLabelerOut2.mat').gTruth;
% [imds, blds] = objectDetectorTrainingData(temp);
% trainingData = combine(imds, blds);

% Temporary Fix
% TODO: make conversion spit out a formattedData with both datastores in it
testData = load('tempTestData.mat').dsnew;
%testData = load('tempTestData.mat').sc;
%testData = load('tempTestData.mat').tc;

% create YOLO v2 object detection network
% TODO: if network dosen't exist run a functionized NetworkMaker.m
lgraph = load('customNetwork.mat').lgraph;

% TODO: if no saved training file, train else use saved training
% or at least something more automatic
doTraining = true
if doTraining
        % Configure the training options. 
    %  * Lower the learning rate to 1e-3 to stabilize training. 
    %  * Set CheckpointPath to save detector checkpoints to a temporary
    %    location. If training is interrupted due to a system failure or
    %    power outage, you can resume training from the saved checkpoint.
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 5, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20);    
    
    % Train YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(trainingData,lgraph,options);
    
    save('trainedDetector.mat', 'detector','info');
else
    pretrained = load('trainedDetector.mat');
    detector = pretrained.detector;
end
testData = trainingData; 

% Create a table to hold the bounding boxes, scores, and labels output by
% the detector. 
numImages = size(testData.UnderlyingDatastores{1,1}.Files, 1);
results = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});

% Run detector on each image in therre test set and collect results.
for i = 1:numImages
    
    % Read the image.
    %I = imread(testData.Var1{i});
    I = imread(testData.UnderlyingDatastores{1,1}.Files{i,1});
    
    % Run the detector.
    [bboxes,scores,labels] = detect(detector,I);
   
    % Collect the results.
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.Labels{i} = labels;
end

% Display sample image with bounding boxes
I = imread(testData.UnderlyingDatastores{1,1}.Files{1,1});
I = insertObjectAnnotation(I, 'rectangle', results.Boxes{1}, results.Scores{1});
imshow(I); 


% TODO: Fix
% Below didn't love being a datastore that's ok for now
% Extract expected bounding box locations from test data.
expectedResults = testData.UnderlyingDatastores{1,2};

% Evaluate the object detector using average precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

