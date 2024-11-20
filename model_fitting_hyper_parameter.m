% Step 1: Set Paths and Load Image Data
dataFolder = './datasets/p_dataset_26';
imageData = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Step 2: Resize Images and Extract HOG Features
inputSize = [64, 64];
cellSize = [16, 16];
hogFeatureSize = length(extractHOGFeatures(ones(inputSize), 'CellSize', cellSize));
numImages = numel(imageData.Files);
features = zeros(numImages, hogFeatureSize, 'single');
labels = imageData.Labels;

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

% Step 5: Hyperparameter Optimization for SVM
% Define hyperparameter grid
boxConstraints = [0.1, 1, 10]; % Values for BoxConstraint
kernelScales = [0.1, 1, 10];   % Values for KernelScale
kernels = {'linear', 'rbf'};   % Kernel functions

% Initialize variables to store results
results = [];

% Loop through each combination of hyperparameters
for bc = boxConstraints
    for ks = kernelScales
        for kf = kernels
            % Train SVM with current hyperparameters
            template = templateSVM('KernelFunction', kf{1}, ...
                                   'BoxConstraint', bc, ...
                                   'KernelScale', ks);
            SVMModel = fitcecoc(trainFeatures, trainLabels, 'Learners', template, 'Coding', 'onevsall');
            
            % Test the SVM model
            predictedLabels = predict(SVMModel, testFeatures);
            accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
            
            % Store the results
            results = [results; {kf{1}, bc, ks, accuracy}];
        end
    end
end

% Step 6: Display Results in a Table
resultsTable = cell2table(results, ...
    'VariableNames', {'KernelFunction', 'BoxConstraint', 'KernelScale', 'Accuracy'});
disp(resultsTable);

% Step 7: Visualize Results in a 3D Plot
figure;
hold on;
uniqueKernels = unique(resultsTable.KernelFunction);
colors = lines(length(uniqueKernels)); % Generate distinct colors for each kernel function

for i = 1:length(uniqueKernels)
    kernel = uniqueKernels{i};
    kernelData = resultsTable(strcmp(resultsTable.KernelFunction, kernel), :);
    scatter3(kernelData.BoxConstraint, kernelData.KernelScale, kernelData.Accuracy, ...
        100, 'filled', 'DisplayName', kernel, 'MarkerFaceColor', colors(i, :));
end

xlabel('BoxConstraint');
ylabel('KernelScale');
zlabel('Accuracy');
title('SVM Hyperparameter Optimization Results');
legend('show');
grid on;
view(45, 30); % Adjust view angle
hold off;

% Highlight Best Result
disp('Best Hyperparameter Combination:');
[~, bestIdx] = max(resultsTable.Accuracy);
bestParameters = resultsTable(bestIdx, :);
disp(bestParameters);

% Annotate the best result on the plot
hold on;
scatter3(bestParameters.BoxConstraint, bestParameters.KernelScale, bestParameters.Accuracy, ...
    200, 'red', 'filled', 'DisplayName', 'Best Result');
text(bestParameters.BoxConstraint, bestParameters.KernelScale, bestParameters.Accuracy, ...
    sprintf('  Accuracy: %.2f%%', bestParameters.Accuracy * 100), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
hold off;


% Step 8: Find and Display Best Parameters
[~, bestIdx] = max(resultsTable.Accuracy);
bestParameters = resultsTable(bestIdx, :);
disp('Best Hyperparameter Combination:');
disp(bestParameters);

% Step 9: Confusion Matrix for Best Model
bestTemplate = templateSVM('KernelFunction', bestParameters.KernelFunction{1}, ...
                           'BoxConstraint', bestParameters.BoxConstraint, ...
                           'KernelScale', bestParameters.KernelScale);
bestSVMModel = fitcecoc(trainFeatures, trainLabels, 'Learners', bestTemplate, 'Coding', 'onevsall');
predictedLabels = predict(bestSVMModel, testFeatures);
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for Best SVM Model');
