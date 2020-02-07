% Post Processor for LP Generator to create an image datastore to train /
% test the detector

clear all 
close all

% Read in raw data from LP generator 
% This should be alpabetical to match images saved in folder
path = "GeneratedImages_2020-02-05_20-10";
rawData = readtable(fullfile(pwd, path, 'dataset.csv'));
rawData = sortrows(rawData);
% Create an Image Datastore
% https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.imagedatastore.html

imds = imageDatastore(fullfile(pwd, path),...
    'IncludeSubfolders', true, 'FileExtensions', '.png');

% Create an Box Label DataStore
numRows = height(rawData);
varNames = {'Boxes', 'Labels'};
varTypes = {'cell', 'cell'};
newTable = table('Size', [numRows, 2], 'VariableNames', varNames, 'VariableTypes', varTypes);
allBoxes = [];

for i = 1:1:numRows
 bboxes = [rawData.Var2(i),rawData.Var3(i),rawData.Var4(i), rawData.Var5(i);...
     rawData.Var6(i), rawData.Var7(i), rawData.Var8(i), rawData.Var9(i);...
     rawData.Var10(i), rawData.Var11(i), rawData.Var12(i), rawData.Var13(i);...
     rawData.Var14(i), rawData.Var15(i), rawData.Var16(i), rawData.Var17(i);...
     rawData.Var18(i), rawData.Var19(i), rawData.Var20(i), rawData.Var21(i);...
     rawData.Var22(i), rawData.Var23(i), rawData.Var24(i), rawData.Var25(i);...
     rawData.Var26(i), rawData.Var27(i), rawData.Var28(i), rawData.Var29(i)];
 
 % TODO: this was a fix for a matlab error - need to solve this in a better
 % way (I can't just have one class)
letters = char(rawData.Var1(i)); 
% labels = [letters(1);
%           string(letters(2));
%           string(letters(3));
%           string(letters(4));
%           string(letters(5));
%           string(letters(6));
%           string(letters(7))];
labels = ["letter";"letter";"letter";"letter";"letter";"letter";"letter";];
newTable(i, 'Boxes') = {bboxes};
newTable(i, 'Labels') = {labels};

allBoxes = vertcat(allBoxes, bboxes');
end

blds = boxLabelDatastore(newTable);


% While I still have the data in a tabular form - let's determine bounding
% boxes
aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
area = prod(allBoxes(:,3:4),2);
figure
scatter(area, aspectRatio)
xlabel("Box Area")
ylabel("Aspect Ratio (width/height)");
title("Box Area vs. Aspect Ratio");
% Number of Anchors is a hyper parameter and needs to be tuned
numAnchors = 4;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(blds, numAnchors);


% merge the 2 datastores
dsnew = combine(imds, blds);


% TODO: a better way to get the test and training data
%save('tempTestData.mat', 'dsnew')
save('formattedData.mat', 'dsnew', 'anchorBoxes')


