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
H3 = fspecial('average',[3,3]);
H5 = fspecial('average',[5,5]);
H7 = fspecial('average',[7,7]);
H9 = fspecial('average',[9,9]);

% Processing of images using filtering
r3 = imfilter(image_rotate,H3);
r5 = imfilter(image_rotate,H5);
r7 = imfilter(image_rotate,H7);
r9 = imfilter(image_rotate,H9);

% Show results
subplot(2,2,1);imshow(image_rotate);title('origin');
subplot(2,2,2);imshow(r3);title('3*3');
subplot(2,2,3);imshow(r5);title('5*5');
subplot(2,2,4);imshow(r7);title('7*7');

% select r7 filter (?)

%-----------------
% Task 1 - Part 4
%----------------- 
        
%I=im2double(I);
I=im2double(r7); % r7 was selected here - might need to revisit
[m,n]=size(I);
M=2*m;N=2*n;%Number of rows and columns of filters
u=-M/2:(M/2-1);
v=-N/2:(N/2-1);
[U,V]=meshgrid(u,v);%The effect of meshgrid(u,v) is to produce two matrices of the same size with vector u as rows and vector v as splits, respectively
D=sqrt(U.^2+V.^2);%Set the distance between the frequency point (U,V) and the center of the frequency domain as D(U,V)
D0=110;             %cut-off frequency
H=double(D>D0);    %Ideal High Pass Filter
J=fftshift(fft2(I,size(H,1),size(H,2))); %Convert time domain image to frequency domain image by Fourier transform and move to the center
K=J.*H;                         %filter processing
L=ifft2(ifftshift(K));          %Fourier inverse transform
L=L(1:m,1:n);                   %Setting the image size
figure;
%subplot(131),imshow(J);
subplot(2,2,1),imshow(I),title("Rotated Image");
subplot(2,2,2),imshow(L),title("High-Pass Filter Applied");
subplot(2,2,3),imhist(I),title("Histogram of Rotated Image");
subplot(2,2,4),imhist(L),title("Histogram after High-Pass Filter")


%-----------------
% Task 1 - Part 5
%----------------- 

% binarized image
BW = imbinarize(L); 
%BW = ~BW; 

% Cleaning up small areas and noise
BW_clean = bwareaopen(BW, 50);% Remove noise less than 50 pixels
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
    [r, c] = find(L == k);
    boundingBox = [min(c), min(r), max(c)-min(c)+1, max(r)-min(r)+1];
    
    % Mark character numbers on the image
    text(boundingBox(1), boundingBox(2)-10, num2str(k), 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
end
title('Marked Characters');

