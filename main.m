clear all;
close all;

%-----------------
%Task 1 - Part 1
%-----------------

% Import image
image = imread("./datasets/HELLO.jpg");

imageR = image(:,:,1);
imageG = image(:,:,2);
imageB = image(:,:,3);

figure;
subplot(2,2,1), imshow(image), axis on;
subplot(2,2,2), imshow(imageR), title("Red Channel");
subplot(2,2,3), imshow(imageG), title("Green Channel");
subplot(2,2,4), imshow(imageB), title("Blue Channel");




%-----------------
% Task 1 - Part 2
%----------------- 

%parameters
imageGray = rgb2gray(image);
modelRotated = imrotate(imageGray,5,'bilinear'); %check using premade function
image_rotate = zeros(size(imageGray),'uint8'); %output

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
                I11 = double(imageGray(up,left));
                I12 = double(imageGray(up,right));
                I21 = double(imageGray(down,left));
                I22 = double(imageGray(down,right));

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




        




