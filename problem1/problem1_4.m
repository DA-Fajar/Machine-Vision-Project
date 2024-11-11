% Read and preprocess the image
I = imread('./HELLO.jpg');
I = rgb2gray(I);
I = im2double(I);

[m, n] = size(I);
M = 2 * m;  % Height (rows)
N = 2 * n;  % Width (columns)

% Generate frequency coordinate grid to match M x N
u = -N/2 : (N/2 - 1); % Frequency coordinates in the width direction
v = -M/2 : (M/2 - 1); % Frequency coordinates in the height direction
[U, V] = meshgrid(u, v);
D = sqrt(U.^2 + V.^2);  % Calculate the distance from each point to the center in the frequency domain

% Define the Ideal High-Pass Filter
D0 = 40;                % Cut-off frequency
H_ideal = double(D > D0); % Ideal high-pass filter mask

% Perform Fourier Transform on the image and pad to M x N
I_fft_shifted = fftshift(fft2(I, M, N)); % Fourier transform and shift zero frequency to center

% Apply the Ideal High-Pass Filter
I_ideal_filtered = I_fft_shifted .* H_ideal; % Apply the Ideal High-Pass Filter
I_ideal_result = ifft2(ifftshift(I_ideal_filtered)); % Inverse Fourier transform
I_ideal_result = real(I_ideal_result(1:m, 1:n)); % Extract real part and crop to original size

% Define the Butterworth High-Pass Filter
Dh = 40; % Cut-off frequency
n_order = 1; % Filter order
H_butter = butter_hp_kernel(I, Dh, n_order); % Generate Butterworth filter kernel

% Check values of H_butter
disp("Max value of H_butter:");
disp(max(H_butter(:)));
disp("Min value of H_butter:");
disp(min(H_butter(:)));

% Display Butterworth filter mask
figure;
imshow(H_butter, []);
title("Butterworth High-Pass Filter Mask");

% Apply the Butterworth High-Pass Filter
I_butter_filtered = I_fft_shifted .* H_butter; % Apply the Butterworth filter

% Check the size of the filtered result
disp("Size of I_butter_filtered:");
disp(size(I_butter_filtered)); % Should be M x N

I_butter_result = ifft2(ifftshift(I_butter_filtered)); % Inverse Fourier transform
I_butter_result = real(I_butter_result(1:m, 1:n)); % Extract real part and crop to original size

% Check values of I_butter_result
disp("Max value of I_butter_result:");
disp(max(I_butter_result(:)));
disp("Min value of I_butter_result:");
disp(min(I_butter_result(:)));

% Display results
figure;
subplot(3, 3, 1), imshow(I), title("Original Image");
subplot(3, 3, 2), imshow(log(1 + abs(I_fft_shifted)), []), title("Frequency Spectrum (Original)");
subplot(3, 3, 3), imshow(I_ideal_result, []), title("Ideal High-Pass Filtered Image");
subplot(3, 3, 4), imhist(I), title("Histogram of Original Image");
subplot(3, 3, 5), imshow(I_butter_result, []), title("Butterworth High-Pass Filtered Image");
subplot(3, 3, 6), imhist(I_ideal_result), title("Histogram after Ideal High-Pass Filter");
subplot(3, 3, 7), imhist(I_butter_result), title("Histogram after Butterworth High-Pass Filter");

% Function Definitions
% Butterworth High-Pass Filter frequency response
function out = butter_hp_f(u, v, Dh, n_order)
    uv = u.^2 + v.^2;
    Duv = sqrt(uv) + eps; % Avoid division by zero
    frac = Dh ./ Duv;
    out = 1 ./ (1 + (frac .^ (2 * n_order)));
end

% Generate Butterworth High-Pass Filter kernel
function out = butter_hp_kernel(I, Dh, n_order) 
    [Height, Width] = size(I); 
    M = 2 * Height;  % Ensure the filter matches the padded Fourier transform size
    N = 2 * Width;

    [u, v] = meshgrid( ...
                    -floor(N/2):floor(N/2)-1, ...
                    -floor(M/2):floor(M/2)-1 ...
                 ); 
    out = butter_hp_f(u, v, Dh, n_order);
end
