% Read the image
Image = imread('./HELLO.jpg');
Image = im2double(rgb2gray(Image)); % Convert to grayscale and double for better filter results

% Define different averaging filters
H3 = fspecial('average', [3, 3]); % 3x3 filter
H5 = fspecial('average', [5, 5]); % 5x5 filter
H7 = fspecial('average', [7, 7]); % 7x7 filter
H9 = fspecial('average', [9, 9]); % 9x9 filter

% Apply the filters to the image
r3 = imfilter(Image, H3);
r5 = imfilter(Image, H5); % This is the 5x5 averaging filter required
r7 = imfilter(Image, H7);
r9 = imfilter(Image, H9);

% Display the results for comparison
figure;
subplot(2, 3, 1); imshow(Image); title('Original Image');
subplot(2, 3, 2); imshow(r3); title('3x3 Mean Filter');
subplot(2, 3, 3); imshow(r5); title('5x5 Mean Filter');
subplot(2, 3, 4); imshow(r7); title('7x7 Mean Filter');
subplot(2, 3, 5); imshow(r9); title('9x9 Mean Filter');
