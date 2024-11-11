% % Read image
% I = imread('./HELLO.jpg');
% figure;
% imshow(I);
% title('original image');

% % Convert to grayscale image
% Igray = rgb2gray(I); 
% figure;
% imshow(Igray);
% title('gray-scale image');

% % binarized image
% BW = imbinarize(Igray); 
% BW = ~BW; 
% figure;
% imshow(BW);
% title('binarized image');

% % Cleaning up small areas and noise
% BW_clean = bwareaopen(BW, 50);% Remove noise less than 50 pixels
% figure;
% imshow(BW_clean);
% title('Binary image after denoising');

% % Detect character boundaries and perform character segmentation
% [L, num] = bwlabel(BW_clean); % Connectivity domain labeling for binary images
% figure;
% imshow(label2rgb(L));% Displays each connectivity field in a different color
% title('Character segmentation results');

% % Extract each character and mark it
% for k = 1:num
%      % Get the bounding box for each character
%     [r, c] = find(L == k); % Get the pixel point of the current character
%     boundingBox = [min(c), min(r), max(c)-min(c)+1, max(r)-min(r)+1];% Determining the bounding box

%      % Draw rectangular boxes on the original image
%     figure;
%     imshow(I);
%     hold on;
%     rectangle('Position', boundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
%     title(['character ', num2str(k), ' bounding box']);
% end

% % Character Recognition with OCR
% results = ocr(BW_clean); % Recognizing Characters Using MATLAB's Built-in OCR Functions
% disp('Recognized text:');
% disp(results.Text); 

% % Displaying images of marked characters
% figure;
% imshow(I);
% hold on;
% for k = 1:num
%     % Get the bounding box for each character
%     [r, c] = find(L == k);
%     boundingBox = [min(c), min(r), max(c)-min(c)+1, max(r)-min(r)+1];
    
%     % Mark character numbers on the image
%     text(boundingBox(1), boundingBox(2)-10, num2str(k), 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
% end
% title('Marked Characters');


% Read and binarize the image
I = imread('HELLO.jpg');
Igray = rgb2gray(I); 
BW = imbinarize(Igray);
BW = ~BW; 

% Remove small noise
BW_clean = bwareaopen(BW, 50); % Remove noise less than 50 pixels
figure;
imshow(BW_clean);
title('Binary Image after Denoising');

% Detect character boundaries and perform character segmentation
[L, num1] = bwlabel(BW_clean); % Connectivity domain labeling for binary images

% Merge regions 6 and 7 into one region
L(L == 7) = 6; % Set all pixels with label 7 to label 6

% Update number of regions since we've merged two regions
unique_labels = unique(L);
unique_labels(unique_labels == 0) = []; % Remove background label (0)
num = length(unique_labels);

% Display the result after merging regions
figure;
imshow(label2rgb(L)); % Display each connectivity field in a different color
title('Character Segmentation Results with Merged Regions');

% Extract each character and mark it
figure;
imshow(I);
hold on;
for k = 1:num
    % Get the bounding box for each character
    [r, c] = find(L == unique_labels(k)); % Get the pixel points of the current character
    boundingBox = [min(c), min(r), max(c)-min(c)+1, max(r)-min(r)+1]; % Determine bounding box

    % Draw rectangular boxes around each character
    rectangle('Position', boundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
    % Display the component number in blue above each character
    text(boundingBox(1), boundingBox(2)-10, num2str(unique_labels(k)), 'Color', 'blue', 'FontSize', 10, 'FontWeight', 'bold');
end
title('Character Bounding Boxes with Labels');

% Character Recognition with OCR
results = ocr(BW_clean); % Recognizing Characters Using MATLAB's Built-in OCR Functions
disp('Recognized text:');
disp(results.Text); 

% Displaying images of marked characters
figure;
imshow(I);
hold on;
for k = 1:num
    % Get the bounding box for each character
    [r, c] = find(L == unique_labels(k));
    boundingBox = [min(c), min(r), max(c)-min(c)+1, max(r)-min(r)+1];
    
    % Mark character numbers on the image
    text(boundingBox(1), boundingBox(2)-10, num2str(unique_labels(k)), 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
end
title('Marked Characters');
