function [outputData] = readImagesIntoDatastoreBB_Fast(imageFile, image_box_map)
% % function [outputData] = readImagesIntoDatastoreBB(imageFile, imageNames, boundingBox)
% This helper function defines the ReadFCN function for image datastores
% related to the CUB_200_2011 dataset.
% It takes a file name of an image file as input.
% We also provide a map from the file name to the related bounding box.
% Authors: Roland Goecke and James Ireland
% Date created: 02/05/22

% Read image file
if isfile(imageFile)
    outputData = imread(imageFile);
else
    disp(imageFile)
end

% Check if image is RGB or grayscal
if size(outputData, 3) < 3
    outputData = cat(3, outputData, outputData, outputData);
end

Filename = split(imageFile, "/");
Filename = split(Filename, "\"); 
xywh_BB = image_box_map(Filename{end}); 

x = xywh_BB(1);
y = xywh_BB(2);
w = xywh_BB(3);
h = xywh_BB(4);

if x > size(outputData, 2) | y > size(outputData, 1)   
        disp("error")
        disp([index, x, y, w, h])
else
    outputData = imcrop(outputData, [x, y, w, h]);
end 
 
end