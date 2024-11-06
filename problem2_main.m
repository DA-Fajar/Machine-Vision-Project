close all;
clear all;

% Load and display the original image
Image = imread('./datasets/hello_world.jpg');
imshow(Image);
title('Original Image');
imageGray = rgb2gray(Image);

% Crop the image to get the middle line (manually specify coordinates)
%J, rect] = imcrop(imageGray);
%subImage = imcrop(imageGray, J);

subImage = imcrop(Image, [1, 99, size(Image,2)/2, 70]); % Replace with actual values
figure;
imshow(subImage);
title('Sub-Image of Middle Line: "HELLO, WORLD!"');

% Convert to grayscale 
% CHAPTER 6 PAGE 5-11
grayImage = rgb2gray(subImage);
figure;
subplot(2,1,1),imshow(grayImage);
subplot(2,1,2),imhist(grayImage);



% Apply thresholding to create a binary image
% binaryImage = imbinarize(grayImage); % thresholding test
binaryImage = zeros(size(grayImage));
for i = 1:size(grayImage,1)
    for j = 1:size(grayImage,2)
        if grayImage(i,j) >= 130 % threshold
            binaryImage(i,j) = 1;
        end
    end
end
figure;
imshow(binaryImage);
title('Binary Image of "HELLO, WORLD!"');

% Create a one-pixel-thin version of the characters
thinImage = bwmorph(binaryImage, 'thin', Inf);
figure;
imshow(thinImage);
title('One-Pixel-Thin Image of Characters');
figure;

outlineImage = bwmorph(binaryImage, 'remove');
figure;
imshow(outlineImage);
title('One-Pixel-Thin Image of Characters');
figure;

% Label connected components
[labeledImage, numCharacters] = bwlabel(outlineImage);
imshow(label2rgb(labeledImage));
title('Segmented and Labeled Characters');

% Display number of characters
disp(['Number of characters: ', num2str(numCharacters)]);

