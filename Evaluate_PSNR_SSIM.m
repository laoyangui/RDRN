function Evaluate_PSNR_SSIM()

clear all; close all; clc

%% set path
degradation = 'BI'; % BI, BD
methods = {'RDRN'};
%dataset = {'Set5','Set14','BSD100','urban100','Manga109'};
dataset = {'Set5'};

ext = {'*.jpg', '*.png', '*.bmp'};
num_method = length(methods);
num_set = length(dataset);
record_results_txt = ['PSNR_SSIM_Results_', degradation,'_model.txt'];
results = fopen(fullfile(record_results_txt), 'wt');

if strcmp(degradation, 'BI') 
    scale_all = [4];
else
    scale_all = 3;
end

for idx_method = 1:num_method
    
    for idx_set = 1:num_set
        fprintf(results, '**********************\n');
        fprintf(results, 'Method_%d: %s; Set: %s\n', idx_method, methods{idx_method}, dataset{idx_set});
        fprintf('**********************\n');
        fprintf('Method_%d: %s; Set: %s\n', idx_method, methods{idx_method}, dataset{idx_set});
        for scale = scale_all
            filepaths = [];
            for idx_ext = 1:length(ext)
                filepaths = cat(1, filepaths, dir(fullfile('./results/HR', dataset{idx_set}, ['x', num2str(scale)], ext{idx_ext})));
            end
            PSNR_all = zeros(1, length(filepaths));
            SSIM_all = zeros(1, length(filepaths));
            for idx_im = 1:length(filepaths)
                name_HR = filepaths(idx_im).name;
                name_SR = strrep(name_HR, 'HR', methods{idx_method});
                im_HR = imread(fullfile('./results/HR', dataset{idx_set}, ['x', num2str(scale)], name_HR));
                im_SR = imread(fullfile('./results/SR', degradation, [methods{idx_method}], dataset{idx_set}, ['x', num2str(scale)], name_SR));
                % change channel for evaluation
                if 3 == size(im_HR, 3)
                    im_HR_YCbCr = single(rgb2ycbcr(im2double(im_HR)));
                    im_HR_Y = im_HR_YCbCr(:,:,1);
                    im_SR_YCbCr = single(rgb2ycbcr(im2double(im_SR)));
                    im_SR_Y = im_SR_YCbCr(:,:,1);
                else
                    im_HR_Y = single(im2double(im_HR));
                    im_SR_Y = single(im2double(im_SR));
                end
                % calculate PSNR, SSIM
                [PSNR_all(idx_im), SSIM_all(idx_im)] = Cal_Y_PSNRSSIM(im_HR_Y*255, im_SR_Y*255, scale, scale);
                fprintf(results, 'x%d %d %s: PSNR= %f SSIM= %f\n', scale, idx_im, name_SR, PSNR_all(idx_im), SSIM_all(idx_im));
                fprintf('x%d %d %s: PSNR= %f SSIM= %f\n', scale, idx_im, name_SR, PSNR_all(idx_im), SSIM_all(idx_im));
            end
            fprintf(results, '--------Mean--------\n');
            fprintf('--------Mean--------\n');
            fprintf(results, 'x%d: PSNR= %f SSIM= %f\n', scale, mean(PSNR_all), mean(SSIM_all));
            fprintf('x%d: PSNR= %f SSIM= %f\n', scale, mean(PSNR_all), mean(SSIM_all));
        end
    end
end
fclose(results);

end

function [psnr_cur, ssim_cur] = Cal_Y_PSNRSSIM(A,B,row,col)
% shave border if needed
if nargin > 2
    [n,m,~]=size(A);
    A = A(row+1:n-row,col+1:m-col,:);
    B = B(row+1:n-row,col+1:m-col,:);
end
% RGB --> YCbCr
if 3 == size(A, 3)
    A = rgb2ycbcr(A);
    A = A(:,:,1);
end
if 3 == size(B, 3)
    B = rgb2ycbcr(B);
    B = B(:,:,1);
end
% calculate PSNR
A=double(A); % Ground-truth
B=double(B); %

e=A(:)-B(:);
mse=mean(e.^2);
psnr_cur=10*log10(255^2/mse);

% calculate SSIM
[ssim_cur, ~] = ssim_index(A, B);
end


function [mssim, ssim_map] = ssim_index(img1, img2, K, window, L)

if (nargin < 2 || nargin > 5)
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

if (size(img1) ~= size(img2))
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

[M N] = size(img1);

if (nargin == 2)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);	%
    K(1) = 0.01;								      % default settings
    K(2) = 0.03;								      %
    L = 255;                                  %
end

if (nargin == 3)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 4)
    [H W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 5)
    [H W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
    ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
    numerator1 = 2*mu1_mu2 + C1;
    numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
    denominator2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(mu1));
    index = (denominator1.*denominator2 > 0);
    ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    index = (denominator1 ~= 0) & (denominator2 == 0);
    ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end