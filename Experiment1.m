clear all;
close all;
clc;

% Week 12 example code for training a simple CNN classifier on
% CUB_200_2011_Subset20classes dataset.
%
% Author: Roland Goecke
% Date created: 26/04/2022
% Modified by: Terence Lam
% Last modified: 08/05/2022

close all;
clear all;
clc;

%% Read the training, validation and test partitions from the relevant
%  text files. 
%  *** Adjust the file path as required. ***
%folder = "P:\assignment\CUB_200_2011_Subset20classes\";
%folder = "P:\assignment\CUB_200_2011\";
folder = "C:\Users\WSLam\Documents\uni canberra\CVIA\assignment1\CUB_200_2011_Subset20classes\";
trainingImageNames = readtable(fullfile(folder, "train.txt"), ... 
    'ReadVariableNames', false);
trainingImageNames.Properties.VariableNames = {'index', 'imageName'};

validationImageNames = readtable(folder + "validate.txt", ... 
    'ReadVariableNames', false);
validationImageNames.Properties.VariableNames = {'index', 'imageName'};

testImageNames = readtable(folder + "test.txt", ... 
    'ReadVariableNames', false);
testImageNames.Properties.VariableNames = {'index', 'imageName'};

%% Read class info from the relevant text files
classNames = readtable(folder + "classes.txt", ...
    'ReadVariableNames', false);
classNames.Properties.VariableNames = {'index', 'className'};

imageClassLabels = readtable(folder + "image_class_labels.txt", ...
    'ReadVariableNames', false);
imageClassLabels.Properties.VariableNames = {'index', 'classLabel'};

%% Create lists of image names for training, validation and test subsets.
%  To be precise, we create an array of strings containing the full file
%  path and file names for each data partition.
trainingImageList = strings(height(trainingImageNames), 1);
for iI = 1:height(trainingImageNames)
    trainingImageList(iI) = string(fullfile(folder, "images/", ...
        string(cell2mat(trainingImageNames.imageName(iI)))));
end

validationImageList = strings(height(validationImageNames), 1);
for iI = 1:height(validationImageNames)
    validationImageList(iI) = string(folder + "images/" + ...
        string(cell2mat(validationImageNames.imageName(iI))));
end

testImageList = strings(height(testImageNames), 1);
for iI = 1:height(testImageNames)
    testImageList(iI) = string(folder + "images/" + ...
        string(cell2mat(testImageNames.imageName(iI))));
end

%% Create image datastores for training, validation and test subsets
trainingImageDS = imageDatastore(trainingImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
trainingImageDS.ReadFcn = @readImagesIntoDatastore;
disp('Training set class distribution:');
countEachLabel(trainingImageDS)

validationImageDS = imageDatastore(validationImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
validationImageDS.ReadFcn = @readImagesIntoDatastore;
disp('Validation set class distribution:');
countEachLabel(validationImageDS)

testImageDS = imageDatastore(testImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
testImageDS.ReadFcn = @readImagesIntoDatastore;
disp('Test set class distribution:');
countEachLabel(testImageDS)

%% The images all have different spatial resolutions (width x height)
targetSize = [224, 224];
trainingImageDS_Resized = transform(trainingImageDS, @(x) imresize(x,targetSize));
validationImageDS_Resized = transform(validationImageDS, @(x) imresize(x,targetSize));
testImageDS_Resized = transform(testImageDS, @(x) imresize(x,targetSize));

% Combine transformed datastores and labels
labelsTraining = arrayDatastore(trainingImageDS.Labels);
cdsTraining = combine(trainingImageDS_Resized, labelsTraining);
labelsValidation = arrayDatastore(validationImageDS.Labels);
cdsValidation = combine(validationImageDS_Resized, labelsValidation);
labelsTest = arrayDatastore(testImageDS.Labels);
cdsTest = combine(testImageDS_Resized, labelsTest);

%%
%img = imread('testimg.png');
%img = imread('Cardinal_0006_17684.jpg');

%% 
% harrisCorners = detectHarrisFeatures(imgGray); 
% imshow(imgGray); hold on; 
% plot(harrisCorners.selectStrongest(100)); 
% hold off;

%% 
% [featureVector,hogVisualization] = extractHOGFeatures(imgGray,'BlockSize',[2 2]);
% %[featureVector,hogVisualization] = extractHOGFeatures(imgGray);
% imshow(imgGray); 
% hold on;
% plot(hogVisualization);

%%
% points = detectSIFTFeatures(imgGray);
% imshow(imgGray);
% hold on;
% plot(points.selectStrongest(50))

%%
% points = detectSURFFeatures(imgGray);
% imshow(imgGray); hold on;
% plot(points.selectStrongest(50));
