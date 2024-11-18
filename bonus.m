% Clear workspace and close all figures
close all;
clear all;

% Step 1: Set Paths and Load Image Data
dataFolder = './datasets/p_dataset_26';
imageData = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Step 2: Resize Images to Consistent Size (e.g., 64x64)
inputSize = [100, 100];
cellSize = [15, 15];
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
Image = imread('./datasets/HELLO.jpg');
height = size(Image,1);
width = size(Image,2);

grayImage = rgb2gray(Image);


% Image rotation

% theta = 4;
% subImage = imrotate(grayImage,theta,'bicubic');
% mask = true(size(grayImage));
% maskR = ~imrotate(mask, theta, 'crop');
% subImage(maskR) = 256;

subImage = zeros(size(grayImage),'uint8'); % memory

nRow = size(subImage,1);
nCol = size(subImage,2);
half_row = floor(nRow / 2);
half_col = floor(nCol / 2);

theta = 4 * pi / 180;
Affine = [
    cos(theta), -sin(theta), 0;
    sin(theta), cos(theta), 0;
    0, 0, 1;
    ];

for y=1:nRow
    for x=1:nCol

        %Coordinatess with respect to center as origin
        xt = x-half_row;
        yt = y-half_col;

        %conducting the rotation, getting new coordinates
        new_coords = Affine*[xt;yt;1];

        %back to original coordinate system
        xn = new_coords(1) + half_row;
        yn = new_coords(2) + half_col;

        %Bilinear Interpolation
        if xn >= 1 && xn <= nCol && yn >= 1 && yn <= nRow

            left = floor(xn);
            right = ceil(xn);
            up = floor(yn);
            down = ceil(yn);

            dx = xn - left;
            dy = yn - up;

            if left >= 1 && right <= nCol && up >= 1 && down <= nRow
                I11 = double(grayImage(up,left));
                I12 = double(grayImage(up,right));
                I21 = double(grayImage(down,left));
                I22 = double(grayImage(down,right));

                Intentsity = (1 - dx) * (1 - dy) * I11 + ...
                             dx * (1 - dy) * I12 + ...
                             (1 - dx) * dy * I21 + ...
                             dx * dy * I22;
                subImage(y,x) = double(Intentsity);

            end
        else 
           subImage(y,x) = 255;
           %let pixel be white
        end
    end
end


% Apply adaptive thresholding to create a binary image
BW = imbinarize(subImage);
BW = ~BW; 

% Remove small noise
binaryImage = bwareaopen(BW, 50); % Remove noise less than 50 pixels

% Create a one-pixel-thin version of the characters
outlineImage = bwmorph(binaryImage, 'remove');

% Label connected components
[labeledImage, numCharacters] = bwlabel(outlineImage);
labeledImage(labeledImage == 3) = 2;
labeledImage(labeledImage == 7) = 6;
labeledImage(labeledImage == 8) = 0;
labeledImage(labeledImage == 9) = 0;
props = regionprops(labeledImage, 'BoundingBox', 'Area');


% Filter out small regions (e.g., punctuation marks)
minArea = 500; % Set a minimum area threshold to filter out small components
validProps = props([props.Area] > minArea);
numValidCharacters = numel(validProps);

% Step 9: Recognize characters in the processed image and display each with a label
recognizedText = '';
figure;
for k = 1:numValidCharacters
    % Extract character region
    bbox = validProps(k).BoundingBox;
    charImage = imcrop(binaryImage, bbox);
    
    % Resize character image while maintaining aspect ratio and pad to input size
    [charHeight, charWidth] = size(charImage);
    scale = min(inputSize(1) / charHeight, inputSize(2) / charWidth);
    newHeight = round(charHeight * scale);
    newWidth = round(charWidth * scale);
    resizedCharImage = imresize(charImage, [newHeight, newWidth]);
    paddedCharImage = zeros(inputSize);
    padTop = floor((inputSize(1) - newHeight) / 2);
    padLeft = floor((inputSize(2) - newWidth) / 2);
    paddedCharImage(padTop + 1 : padTop + newHeight, padLeft + 1 : padLeft + newWidth) = resizedCharImage;
    resizedCharImage = paddedCharImage;
    
    % Perform morphological operations to refine the character
    se = strel('disk', 1);
    refinedCharImage = imclose(resizedCharImage, se);
    
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
    imshow(refinedCharImage);
    title(sprintf('Pred: %s', predictedLabel));
end

% Display the recognized text in console
fprintf('Recognized Text: %s\n', recognizedText);

% Display the recognized text in console
fprintf('Recognized Text: %s\n', recognizedText);