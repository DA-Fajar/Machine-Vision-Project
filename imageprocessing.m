% Step 1: Load and Crop the Original Image
Image = imread('./datasets/hello_world.jpg'); % Load the image
subImage = imcrop(Image, [1, 99, size(Image,2)/2, 70]); % Crop to middle line
grayImage = rgb2gray(subImage); % Convert to grayscale

% Display the cropped grayscale image
figure;
imshow(grayImage);
title('Cropped Grayscale Image');

% Step 2: Adaptive Thresholding to Create Binary Image
binaryImage = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.35); % Adjust sensitivity if needed

% Display the binary image
figure;
imshow(binaryImage);
title('Binary Image (Adaptive Thresholding)');

% Step 3: Apply Morphological Operations for Smoothing
se = strel('disk', 0); 
smoothedBinaryImage = imclose(imopen(binaryImage, se), se); % Open and close operations

% Display the smoothed binary image
figure;
imshow(smoothedBinaryImage);
title('Smoothed Binary Image');

% Step 4: Generate One-Pixel-Wide Character Outlines
outlineImage = bwmorph(smoothedBinaryImage, 'remove'); % Extract one-pixel-wide outlines

% Display original binary and one-pixel thin images for comparison
figure;
subplot(1, 2, 1);
imshow(smoothedBinaryImage);
title('Original Binary Image');

subplot(1, 2, 2);
imshow(outlineImage);
title('One-Pixel Thin Image');

% Step 5: Label Connected Regions and Extract Properties
[labeledImage, numCharacters] = bwlabel(outlineImage, 8); % Label connected regions
props = regionprops(labeledImage, 'BoundingBox', 'Area'); % Extract region properties

% Step 6: Filter Small Regions (e.g., Noise and Punctuation Marks)
minArea = 100; % Minimum area threshold
validProps = props([props.Area] > minArea); % Filter out small regions

% Step 7: Visualize Filtered Regions
figure;
imshow(outlineImage);
hold on;
for k = 1:numel(validProps)
    rectangle('Position', validProps(k).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2); % Draw bounding boxes
end
title('Filtered Character Regions');

% Step 8: Visualize Unfiltered Regions for Comparison
figure;
imshow(outlineImage);
title('Unfiltered Character Regions (No Area Filtering)');

% Step 9: Draw Bounding Boxes on the Original Image
figure;
imshow(grayImage);
hold on;
for k = 1:numel(validProps)
    rectangle('Position', validProps(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 2); % Draw bounding boxes on original image
end
title('Bounding Boxes on Original Image');
