% End-to-End Network Developer
% Christina Paolicelli
% 2/10/20

% ---- High Level Script ---- %

% TODO: make path and "new" conversion / "new" network scripted rather than
% manually modified
TrainingPath = "GeneratedImages_2020-02-21_10-41";
TestPath = "GeneratedImages_2020-02-21_10-41";
doConversion = true; 
makeNetwork = false;
doTraining = true; 

% TODO: figure out scheme of saving between sessions in a programatic way
if(doConversion)
    [trainingData, anchorBoxes] = Conversion(TrainingPath);
    [testData, ~] = Conversion(TestPath); 
    save('dataset.mat', 'trainingData', 'testData', 'anchorBoxes');
else
    trainingData = load('dataset.mat').trainingData;
    testData = load('dataset.mat').testData;
    anchorBoxes = load('dataset.mat').anchorBoxes;
end


if(makeNetwork)
    [network] = NetworkMaker(anchorBoxes);
    save('network.mat', 'network');
else
    network = load('network.mat').network;
end

if(doTraining)
    [detector, info] = TrainDetector(trainingData, network);
    
    % Generate figures of training data
    figure
    plot(info.TrainingLoss)
    grid on
    xlabel('Number of Iterations')
    ylabel('Training Loss for Each Iteration')
    
    save('training.mat', 'detector', 'info');
else
    detector = load('training.mat').detector;
end
% TODO: else load pretrained detector

results = TestDetector(testData, detector);
expectedResults = testData.UnderlyingDatastores{1,2};
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

% Generate Plot of results
% I don't have any results so it fails here for me
figure
plot([recall{:}], [precision{:}])
xlabel('Recall')
ylabel('Percision')
grid on
%title(sprintf('Average Precision = %.2f',ap))


% Evaluate Results
function [results] = TestDetector(testData, detector)
    % Create a table to hold the bounding boxes, scores, and labels output by
    % the detector. 
    numImages = size(testData.UnderlyingDatastores{1,1}.Files, 1);
    results = table('Size',[numImages 3],...
        'VariableTypes',{'cell','cell','cell'},...
        'VariableNames',{'Boxes','Scores','Labels'});
    
    
    % Run detector on each image in therre test set and collect results.
    for i = 1:numImages

        % Read the image.
        I = imread(testData.UnderlyingDatastores{1,1}.Files{i,1});

        % Run the detector.
        [bboxes,scores,labels] = detect(detector,I);

        % Collect the results.
        results.Boxes{i} = bboxes;
        results.Scores{i} = scores;
        results.Labels{i} = labels;
    end
    
end

% Trains Detector
function [detector, info] = TrainDetector(trainingData, network)
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 8, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
	'ExecutionEnvironment','gpu');
    [detector,info] = trainYOLOv2ObjectDetector(trainingData,network,options);
end

% Creates a Network to use to train and test
function [network] = NetworkMaker(anchorBoxes)
    % Input Size - this is specific to my network !
    imageInputSize = [256, 512, 3];
    
    % Create a new input image layer
    % Set the name to the original layer name
    imgLayer = imageInputLayer(imageInputSize, "Name", "input_1");
    
    % load manually modified mobileNetv2 network
    % TODO: try other networks based on papers
    modified = load('modifiedMobileNetv2.mat');
    lgraph = modified.lgraph_1;
    
    % Replace old image input layer
    lgraph = replaceLayer(lgraph, "input_1", imgLayer);
    
    % Create YOLOv2 Detection SubNetwork
    
    % Set the convolution layer filter size to [3 3]
    % This size is common in CNN architectures
    filterSize = [3,3];
    
    % Set the number of filters in the convolution layers to match the number
    % of channels in the feature extrraction layer output
    % This varies with the number of classes and anchor boxes
    numFilters = 74; 
    
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

    numClasses = 32;
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
    
    % so this layer is [19,19] times 16 this is about 300
    % because we have a 2:1 ratio this is kinda inbetween
    % and since this is used in the tutorial I'm sticking
    % with it for now
    featureExtractionLayer = "block_12_add";
    
    % Connect the detection subnetwork to the feature extraction layer.
    lgraph = connectLayers(lgraph,featureExtractionLayer,"yolov2Conv1");

    network = lgraph;
end

% Takes in path to csv and spits out ground truth datastore & anchor boxes
function [dsnew, anchorBoxes] = Conversion(apath)
    % Assummes Data has been generated with Python LPImageGenerator tool
    % path to generated data is passed in

    % Read in rawdata
    fullfile(pwd, apath, 'dataset.csv')
    rawData = readtable(fullfile(pwd, apath, 'dataset.csv'));

    % Sort alphabetically to match file order
    rawData = sortrows(rawData);

    % Create an imagedatastore
    imds = imageDatastore(fullfile(pwd, apath),...
        'IncludeSubfolders', true, 'FileExtensions', '.png');

    % Create a Box Label Datastore
    size(rawData)
    numRows = height(rawData)
    varNames = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',...
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};
    varTypes = {'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell',...
        'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell',...
        'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell'};
    newTable = table('Size', [numRows, 32], 'VariableNames', varNames, 'VariableTypes', varTypes);


    for i = 1:1:numRows
        % Populate the table
        letters = char(rawData.Var1(i)); 
        row = {[rawData.Var2(i), rawData.Var3(i), rawData.Var4(i), rawData.Var5(i)]};
        newTable(i, letters(1)) = {row};
        row = {[rawData.Var6(i), rawData.Var7(i), rawData.Var8(i), rawData.Var9(i)]};
        newTable(i, letters(2)) = {row};
        row = {[rawData.Var10(i), rawData.Var11(i), rawData.Var12(i), rawData.Var13(i)]};
        newTable(i, letters(3)) = {row};
        row = {[rawData.Var14(i), rawData.Var15(i), rawData.Var16(i), rawData.Var17(i)]};
        newTable(i, letters(4)) = {row};
        row = {[rawData.Var18(i), rawData.Var19(i), rawData.Var20(i), rawData.Var21(i)]};
        newTable(i, letters(5)) = {row};
        row = {[rawData.Var22(i), rawData.Var23(i), rawData.Var24(i), rawData.Var25(i)]};
        newTable(i, letters(6)) = {row};
        row = {[rawData.Var26(i), rawData.Var27(i), rawData.Var28(i), rawData.Var29(i)]};
        newTable(i, letters(7)) = {row};
        
    end
    size(newTable); 
    blds = boxLabelDatastore(newTable);
    
    % While we have the data in tabular form determine training data driven
    % anchor boxes
    numAnchors = 2;
    [anchorBoxes, ~] = estimateAnchorBoxes(blds, numAnchors);
    
    % merge the 2 datastores
    dsnew = combine(imds, blds);
end
