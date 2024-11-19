% Clear workspace and close all figures
close all;
clear all;

% Step 1: Set Paths and Load Image Data
dataFolder = './datasets/p_dataset_26';
imageData = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Step 2: Resize Images to Consistent Size (e.g., 64x64)
inputSize = [64, 64];
cellSize = [8, 8];
hogFeatureSize = length(extractHOGFeatures(ones(inputSize), 'CellSize', cellSize));
numImages = numel(imageData.Files);
features = zeros(numImages, hogFeatureSize, 'single');
labels = imageData.Labels;

% Extract HOG features from each image in the dataset
for i = 1:numImages
    img = readimage(imageData, i);
    grayImage = im2gray(img);  % Convert to grayscale
    resizedImage = imresize(grayImage, inputSize);  % Resize to consistent size
    features(i, :) = extractHOGFeatures(resizedImage, 'CellSize', cellSize);  % Extract HOG features
end

% Step 3: Split Data into Training and Testing Sets
[trainIdx, testIdx] = dividerand(numImages, 0.75, 0.25);
trainFeatures = features(trainIdx, :);
trainLabels = labels(trainIdx);
testFeatures = features(testIdx, :);
testLabels = labels(testIdx);

% Step 4: Standardize Features
[trainFeatures, mu, sigma] = zscore(trainFeatures);
testFeatures = (testFeatures - mu) ./ sigma;

% Step 5: Train an SVM Model
SVMModel = fitcecoc(trainFeatures, trainLabels, 'Learners', 'svm', 'Coding', 'onevsall');

% Step 6: Test the Classifier
predictedLabels = predict(SVMModel, testFeatures);
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
fprintf('Classification Accuracy: %.2f%%\n', accuracy * 100);

% Step 7: Display Confusion Matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for Character Classification');

% % Close the figure after displaying the confusion matrix
% pause(2); % Pause to allow viewing of the confusion matrix before it closes
% close(gcf); % Close the confusion matrix figure
%% 

% Step 8: Load and Process the Original Image for Character Recognition
Image = imread('./datasets/hello_world.jpg');

% Crop the image to get the middle line (manually specify coordinates)
subImage = imcrop(Image, [8, 99, size(Image,2)/2, 70]); % Replace with actual values

% Convert to grayscale
grayImage = rgb2gray(subImage);

% Apply adaptive thresholding to create a binary image
binaryImage = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.32); % Adjust sensitivity if necessary

% Create a one-pixel-thin version of the characters
outlineImage = bwmorph(binaryImage, 'remove');

% Label connected components
[labeledImage, numCharacters] = bwlabel(outlineImage);
props = regionprops(labeledImage, 'BoundingBox', 'Area');

% Filter out small regions (e.g., punctuation marks)
minArea = 100; % Set a minimum area threshold to filter out small components
validProps = props([props.Area] > minArea);
numValidCharacters = numel(validProps);

% Step 9: Recognize characters in the processed image and display each with a label
recognizedText = '';
figure;
for k = 1:numValidCharacters
    % Extract character region
    bbox = validProps(k).BoundingBox;
    charImage = imcrop(binaryImage, bbox);
    
    % Resize character image to match input size of SVM model
    resizedCharImage = imresize(charImage, inputSize);
    
    % Perform morphological operations to refine the character
    se = strel('disk', 1); % Adjust structuring element size as needed
    refinedCharImage = imclose(resizedCharImage, se); % Apply morphological closing
    
    % Extract HOG features from the refined image
    charFeatures = extractHOGFeatures(refinedCharImage, 'CellSize', cellSize);
    
    % Standardize features
    standardizedFeatures = (charFeatures - mu) ./ sigma;
    
    % Predict the label using the trained SVM model
    predictedLabel = predict(SVMModel, standardizedFeatures);
    
    predictedLabel = erase(string(predictedLabel), "Sample");

    recognizedText = strcat(recognizedText, predictedLabel);

    % Display each character in a subplot with its cleaned predicted label
    subplot(ceil(sqrt(numValidCharacters)), ceil(sqrt(numValidCharacters)), k);
    imshow(refinedCharImage); % Show refined character
    title(sprintf('Pred: %s', predictedLabel));
end

% Display the recognized text in console
fprintf('Recognized Text: %s\n', recognizedText);
