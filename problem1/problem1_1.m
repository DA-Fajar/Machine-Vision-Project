% Display the original image on screen. Convert the image to grayscale image.
% Experiment with contrast enhancement of the image. Comment on the results
original_image = imread('./HELLO.jpg');
imshow(original_image);
title('Original Image');

grayscale_image = rgb2gray(original_image); %covert the original image to grayscale image
figure;
imshow(grayscale_image);
% method 1
% using image tool to adjust image contrast
imtool(grayscale_image); %#ok<IMTOOL>

% Apply histogram equalization to enhance contrast
equalized_image = histeq(grayscale_image); 
figure; 
imshow(equalized_image);
title('Image after Histogram Equalization');

% Display the histogram of the original grayscale image
figure;
imhist(grayscale_image);
title('Histogram of the Original Grayscale Image');

% Display the histogram of the equalized image
figure;
imhist(equalized_image);
title('Histogram of the Equalized Image');