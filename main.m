clear all;
close all;

%-----------------
%Task 1 - Part 1
%-----------------

% Import image
image = imread("./datasets/HELLO.jpg");
imageGray = rgb2gray(image);
imageR = image(:,:,1);
imageG = image(:,:,2);
imageB = image(:,:,3);

figure;
subplot(2,2,1), imshow(image), axis on;
subplot(2,2,2), imshow(imageR), title("Red Channel");
subplot(2,2,3), imshow(imageG), title("Green Channel");
subplot(2,2,4), imshow(imageB), title("Blue Channel");

grayscale_image = rgb2gray(image); %covert the original image to grayscale image
% method 1
% using image tool to adjust image contrast
% imtool(grayscale_image); %#ok<IMTOOL>
% Apply histogram equalization to enhance contrast
equalized_image = histeq(grayscale_image); 

figure; 
subplot(2,2,1),imshow(grayscale_image),title('Image after greyscale');
subplot(2,2,2),imshow(equalized_image),title('Image after Histogram Equalization');
subplot(2,2,3),imhist(grayscale_image),title('Histogram of the Original Grayscale Image');
subplot(2,2,4),imhist(equalized_image),title('Histogram of the Equalized Image');

% Channel Histogram visualization
freq = zeros(256,3);
for i=1:size(imageR,1)
    for j=1:size(imageR,2)
        freq(imageR(i,j)+1,1) = freq(imageR(i,j)+1,1) + 1; 
        freq(imageG(i,j)+1,2) = freq(imageG(i,j)+1,1) + 1; 
        freq(imageB(i,j)+1,3) = freq(imageB(i,j)+1,1) + 1;
    end
end
figure;
subplot(3,1,1);
bar(0:255,freq(:,1),'r');
title('red channel');
subplot(3,1,2);
bar(0:255,freq(:,2),'g');
title('green channel');
subplot(3,1,3);
bar(0:255,freq(:,3),'b');
title('blue channel');


%-----------------
% Task 1 - Part 2
%----------------- 

%parameters
modelRotated = imrotate(imageGray,5,'bilinear'); %check using premade function
image_rotate = zeros(size(equalized_image),'uint8'); %output

nRow = size(image_rotate,1);
nCol = size(image_rotate,2);
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

%           (1,1)----------(1,2)
%             |              |
%             |      (y,x)   |
%             |              |
%           (2,1)----------(2,2)

            if left >= 1 && right <= nCol && up >= 1 && down <= nRow
                I11 = double(equalized_image(up,left));
                I12 = double(equalized_image(up,right));
                I21 = double(equalized_image(down,left));
                I22 = double(equalized_image(down,right));

                Intentsity = (1 - dx) * (1 - dy) * I11 + ...
                             dx * (1 - dy) * I12 + ...
                             (1 - dx) * dy * I21 + ...
                             dx * dy * I22;
                image_rotate(y,x) = double(Intentsity);

            end
        else 
           image_rotate(y,x) = 255;
           %let pixel be white
        end
    end
end


%plotting rotation results:
figure;
subplot(2,1,2),imshow(image_rotate),title("rotation - bilinear interpolation"), axis on;
subplot(2,1,1),imshow(imageGray),title("Original Greysacle Image"), axis on;


%-----------------
% Task 1 - Part 3
%----------------- 


% Setting the mean value filter
H3 = fspecial("average",[3,3]);
H5 = fspecial('average',[5,5]);
H7 = fspecial('average',[7,7]);
H9 = fspecial('average',[9,9]);

% Processing of images using filtering
r3 = imfilter(image_rotate,H3);
r5 = imfilter(image_rotate,H5);
r7 = imfilter(image_rotate,H7);
r9 = imfilter(image_rotate,H9);

% Show results
subplot(2,2,1);imshow(image_rotate);title('rotated image');
subplot(2,2,2);imshow(r3);title('3*3');
subplot(2,2,3);imshow(r5);title('5*5');
subplot(2,2,4);imshow(r7);title('7*7');

%-----------------
% Task 1 - Part 4
%----------------- 
        

% Second implementation:
% butterworth filter
function out = ifftshow(f)
    f1 = abs(f);
    fm = max(f1(:));
    out = f1/fm;
end

function k = butter_hp_kernel(I, Dh, n) 
    Height = size(I,1); 
    Width = size(I,2); 

    [u, v] = meshgrid( ...
                    -floor(Width/2) :floor(Width-1)/2, ...
                    -floor(Height/2): floor(Height-1)/2 ...
                 ); 

    k = butter_hp_f(u, v, Dh, n);
end

function f = butter_hp_f(u, v, Dh, n)
    uv = u.^2+v.^2;
    Duv = sqrt(uv);
    frac = Dh./Duv;
    %denom = frac.^(2*n);
    A=0.414; denom = A.*(frac.^(2*n));    
    f = 1./(1.+denom);
end

function [out1, out2] = butterworth_hpf(I, Dh, n)

    Kernel = butter_hp_kernel(I, Dh, n);

    I_ffted_shifted = fftshift(fft2(double(I)));

    I_ffted_shifted_filtered = I_ffted_shifted.*Kernel;

    out1 = ifftshow(ifft2(I_ffted_shifted_filtered));

    out2 = ifft2(ifftshift(I_ffted_shifted_filtered));
end

Dh = 40; % threshold
n = 1; % hpf order
[J, butter] = butterworth_hpf(r7, Dh, n);


figure;
%subplot(131),imshow(J);
subplot(2,3,1),imshow(r5),title("5x5Filtered-Rotated Image");
subplot(2,3,2),imshow(J),title("High-Pass Filter Applied");
subplot(2,3,3),imshow(butter),title("5x5Filtered-Rotated Image");
subplot(2,3,4),imhist(r5),title("Histogram of Rotated Image");
subplot(2,3,5),imhist(J),title("Histogram after High-Pass Filter")
subplot(2,3,6),imhist(butter),title("High-Pass Filter Applied");


%-----------------
% Task 1 - Part 5
%----------------- 

% binarized image
T = adaptthresh(butter, 0.9); % Adjust the sensitivity as needed
BW = imbinarize(butter);
BW = ~BW; 
% binarized 

% Cleaning up small areas and noise
BW_clean = bwareaopen(BW, 10000);% Remove noise less than 10000 pixels
figure;
imshow(BW_clean);
title('Binary image after denoising');

% Detect character boundaries and perform character segmentation
[Clean, num] = bwlabel(BW_clean); % Connectivity domain labeling for binary images
figure;
imshow(label2rgb(Clean));% Displays each connectivity field in a different color
title('Character segmentation results');

% Extract each character and mark it
for k = 1:num
     % Get the bounding box for each character
    [r, c] = find(Clean == k); % Get the pixel point of the current character
    boundingBox = [min(c), min(r), max(c)-min(c)+1, max(r)-min(r)+1];% Determining the bounding box
end

% Character Recognition with OCR
results = ocr(BW_clean); % Recognizing Characters Using MATLAB's Built-in OCR Functions
disp('Recognized text:');
disp(results.Text); 

for k = 1:num
    % Get the bounding box for each character
    [r, c] = find(Clean == k);
    boundingBox = [min(c), min(r), max(c)-min(c)+1, max(r)-min(r)+1];
    
    % Mark character numbers on the image
    text(boundingBox(1), boundingBox(2)-10, num2str(k), 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
end
title('Marked Characters');

