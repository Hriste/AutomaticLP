% Post Processor for LP Generator to create an image datastore to train /
% test the detector

clear all 
close all

% Read in raw data from LP generator 
% This should be alpabetical to match images saved in folder
path = "GeneratedImages_2020-02-09_17-10";
rawData = readtable(fullfile(pwd, path, 'dataset.csv'));
rawData = sortrows(rawData);
% Create an Image Datastore
% https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.imagedatastore.html

imds = imageDatastore(fullfile(pwd, path),...
    'IncludeSubfolders', true, 'FileExtensions', '.png');

% Create an Box Label DataStore
numRows = height(rawData);
varNames = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',...
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};
varTypes = {'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell',...
    'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell',...
    'cell', 'cell', 'cell', 'cell', 'cell', 'cell', 'cell'};

newTable = table('Size', [numRows, 32], 'VariableNames', varNames, 'VariableTypes', varTypes);

singleClass = table('Size', [numRows, 1], 'VariableNames', {'char'}, 'VariableTypes', {'cell'});
twoClass = table('Size', [numRows, 2], 'VariableNames', {'letter', 'num'}, 'VariableTypes', {'cell', 'cell'});

allBoxes = [];


for i = 1:1:numRows
 % keeping this around so we can do anchor boxes the same way
 bboxes = [rawData.Var2(i),rawData.Var3(i),rawData.Var4(i), rawData.Var5(i);...
     rawData.Var6(i), rawData.Var7(i), rawData.Var8(i), rawData.Var9(i);...
     rawData.Var10(i), rawData.Var11(i), rawData.Var12(i), rawData.Var13(i);...
     rawData.Var14(i), rawData.Var15(i), rawData.Var16(i), rawData.Var17(i);...
     rawData.Var18(i), rawData.Var19(i), rawData.Var20(i), rawData.Var21(i);...
     rawData.Var22(i), rawData.Var23(i), rawData.Var24(i), rawData.Var25(i);...
     rawData.Var26(i), rawData.Var27(i), rawData.Var28(i), rawData.Var29(i)];
 
    % ok let's now populate the table
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
    
    singleClass(i, 1) = {bboxes};
    twoClass(i,1) = {[rawData.Var6(i), rawData.Var7(i), rawData.Var8(i), rawData.Var9(i); rawData.Var10(i), rawData.Var11(i), rawData.Var12(i), rawData.Var13(i)]};
    twoClass(i,2) = {[rawData.Var2(i), rawData.Var3(i), rawData.Var4(i), rawData.Var5(i);...
        rawData.Var14(i), rawData.Var15(i), rawData.Var16(i), rawData.Var17(i);...
        rawData.Var18(i), rawData.Var19(i), rawData.Var20(i), rawData.Var21(i);...
        rawData.Var22(i), rawData.Var23(i), rawData.Var24(i), rawData.Var25(i);...
        rawData.Var26(i), rawData.Var27(i), rawData.Var28(i), rawData.Var29(i)]};
    
    allBoxes = vertcat(allBoxes, bboxes');
end

blds = boxLabelDatastore(newTable);
blds2 = boxLabelDatastore(singleClass);
blds3 = boxLabelDatastore(twoClass);


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
numAnchors = 2;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(blds, numAnchors);


% merge the 2 datastores
dsnew = combine(imds, blds);
sc = combine(imds, blds2);
tc = combine(imds, blds3);

% TODO: a better way to get the test and training data
%save('tempTestData.mat', 'dsnew', 'sc','tc')
save('formattedData.mat', 'dsnew', 'anchorBoxes', 'sc','tc');


%Figuring out ideal number of anchor boxes
% maxNumAnchors = 15;
% meanIoU = zeros([maxNumAnchors,1]);
% anchorBoxes = cell(maxNumAnchors, 1);
% for k = 1:maxNumAnchors
%     % Estimate anchors and mean IoU.
%     [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(blds,k);    
% end
% 
% figure
% plot(1:maxNumAnchors,meanIoU,'-o')
% ylabel("Mean IoU")
% xlabel("Number of Anchors")
% title("Number of Anchors vs. Mean IoU")


