close all
clear all
% Get image name and bounding box info
data = load('GeneratedImages_2020-01-30_19-08/dataset.mat');
path = "GeneratedImages_2020-01-30_19-08";
%data.field = fullfile(pwd, path, data.field.strsplit("_"));

% Read one of the images
I = imread(fullfile(pwd, path, '3PC9321.png'));
% Insert ROI Labels
I = insertShape(I, 'Rectangle', [data.file_3PC9321(1,1),data.file_3PC9321(1,2), data.file_3PC9321(1,3), data.file_3PC9321(1,4)]);
I = imresize(I, 3);
imshow(I)

% Randomly split data into a training set and a test set
% TODO: eventually make this random for now 60% training 40% testing
% which is... 12/8
trainingData = data

