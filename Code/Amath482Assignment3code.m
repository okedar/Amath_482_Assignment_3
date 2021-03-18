% clear all;  close all;  clc;



%%                                     Tracking
%%
%                                     cam1_1                         1

% close all
load('cam1_1.mat')
% implay(vidFrames1_1)

GrayFrameAll = vidFrames1_1(:,:,:,11:226);
numFrames = size(GrayFrameAll,4);
X_time1_1 = zeros(1,numFrames);
Y_time1_1 = zeros(1,numFrames);

% initial max
x = 323;
y = 222;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+50:end) = 0;
    % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+20:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time1_1(j) = x;
    Y_time1_1(j) = y;
    
%     figure(1);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time1_1(j), Y_time1_1(j), 'bo', 'markersize', 10);
%     drawnow
end

S1(1,:) = X_time1_1;
S1(2,:) = Y_time1_1;
% Ssize = size(S);
% Srow = Ssize(1,1) + 1;
% S(Srow,:) = Y_time1_1;



%%
%                                     cam2_1                         1

% close all
load('cam2_1.mat')
% implay(vidFrames2_1)

GrayFrameAll = vidFrames2_1(:,:,:,20:20+215);
numFrames = size(GrayFrameAll,4);
X_time2_1 = zeros(1,numFrames);
Y_time2_1 = zeros(1,numFrames);

% initial max
x = 277;
y = 111;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+50:end) = 0;
    % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+20:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time2_1(j) = x;
    Y_time2_1(j) = y;
        
%     figure(2);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time2_1(j), Y_time2_1(j), 'bo', 'markersize', 10);
%     drawnow
end

S1(3,:) = X_time2_1;
S1(4,:) = Y_time2_1;




%%
%                                     cam3_1                         1

% close all
load('cam3_1.mat')
% implay(vidFrames3_1)

GrayFrameAll = (vidFrames3_1(:,:,:,11:11+215));
numFrames = size(GrayFrameAll,4);
X_time3_1 = zeros(1,numFrames);
Y_time3_1 = zeros(1,numFrames);

% initial max
x = 286;
y = 262;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
   
        % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+50:end,:) = 0;
        % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+20:end) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time3_1(j) = x;
    Y_time3_1(j) = y;

%     figure(3);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time3_1(j), Y_time3_1(j), 'bo', 'markersize', 10);
%     drawnow
end

S1(5,:) = X_time3_1;
S1(6,:) = Y_time3_1;







%%
%                                     cam1_2                         2

% close all
load('cam1_2.mat')
% implay(vidFrames1_2)

GrayFrameAll = vidFrames1_2(:,:,:,13:end);
numFrames = size(GrayFrameAll,4);
X_time1_2 = zeros(1,numFrames);
Y_time1_2 = zeros(1,numFrames);

% initial max
x = 332;
y = 332;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+50:end) = 0;
    % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+20:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time1_2(j) = x;
    Y_time1_2(j) = y;
    
%     figure(1);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time1_2(j), Y_time1_2(j), 'bo', 'markersize', 10);
%     drawnow
end

S2(1,:) = X_time1_2;
S2(2,:) = Y_time1_2;




%%
%                                     cam2_2                         2

% close all
load('cam2_2.mat')
% implay(vidFrames2_2)

GrayFrameAll = vidFrames2_2(:,:,:,1:1+301);
numFrames = size(GrayFrameAll,4);
X_time2_2 = zeros(1,numFrames);
Y_time2_2 = zeros(1,numFrames);

% initial max
x = 317;
y = 360;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+50:end) = 0;
    % y filter
    GrayFrame(1:y-75,:) = 0;
    GrayFrame(y+50:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time2_2(j) = x;
    Y_time2_2(j) = y;
    
%     figure(1);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time2_2(j), Y_time2_2(j), 'bo', 'markersize', 10);
%     drawnow
end

S2(3,:) = X_time2_2;
S2(4,:) = Y_time2_2;





%%
%                                     cam3_2                         2

% close all
load('cam3_2.mat')
% implay(vidFrames3_2)

GrayFrameAll = vidFrames3_2(:,:,:,18:18+301);
numFrames = size(GrayFrameAll,4);
X_time3_2 = zeros(1,numFrames);
Y_time3_2 = zeros(1,numFrames);

% initial max
x = 412;
y = 247;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-70) = 0;
    GrayFrame(:,x+30:end) = 0;
    % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+50:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time3_2(j) = x;
    Y_time3_2(j) = y;
    
%     figure(1);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time3_2(j), Y_time3_2(j), 'bo', 'markersize', 10);
%     drawnow
end

S2(5,:) = X_time3_2;
S2(6,:) = Y_time3_2;






%%
%                                     cam1_3                         3

%close all
load('cam1_3.mat')
% implay(vidFrames1_3)

GrayFrameAll = vidFrames1_3(:,:,:,18:18+221);
numFrames = size(GrayFrameAll,4);
X_time1_3 = zeros(1,numFrames);
Y_time1_3 = zeros(1,numFrames);

% initial max
x = 301;
y = 317;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+50:end) = 0;
    % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+20:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time1_3(j) = x;
    Y_time1_3(j) = y;
    
%     figure(1);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time1_3(j), Y_time1_3(j), 'bo', 'markersize', 10);
%     drawnow
end

S3(1,:) = X_time1_3;
S3(2,:) = Y_time1_3;



%%
%                                     cam2_3                         3

% close all
load('cam2_3.mat')
% implay(vidFrames2_3)

GrayFrameAll = vidFrames2_3(:,:,:,44:44+221);
numFrames = size(GrayFrameAll,4);
X_time2_3 = zeros(1,numFrames);
Y_time2_3 = zeros(1,numFrames);

% initial max
x = 267;
y = 295;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-20) = 0;
    GrayFrame(:,x+20:end) = 0;
    % y filter
    GrayFrame(1:y-60,:) = 0;
    GrayFrame(y+15:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time2_3(j) = x;
    Y_time2_3(j) = y;
    
%     figure(1);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time2_3(j), Y_time2_3(j), 'bo', 'markersize', 10);
%     drawnow
end

S3(3,:) = X_time2_3;
S3(4,:) = Y_time2_3;




%%
%                                     cam3_3                         3

% close all
load('cam3_3.mat')
% implay(vidFrames3_3)

GrayFrameAll = vidFrames3_3(:,:,:,9:9+221);
numFrames = size(GrayFrameAll,4);
X_time3_3 = zeros(1,numFrames);
Y_time3_3 = zeros(1,numFrames);

% initial max
x = 379;
y = 272;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
    
    % x filter
    GrayFrame(:,1:x-70) = 0;
    GrayFrame(:,x+30:end) = 0;
    % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+50:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time3_3(j) = x;
    Y_time3_3(j) = y;
    
%     figure(1);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time3_3(j), Y_time3_3(j), 'bo', 'markersize', 10);
%     drawnow
end

S3(5,:) = X_time3_3;
S3(6,:) = Y_time3_3;



%%
%                                     cam1_4                         4

% close all
load('cam1_4.mat')
% implay(vidFrames1_4)

GrayFrameAll = (vidFrames1_4(:,:,:,8:8+384));
numFrames = size(GrayFrameAll,4);
X_time1_4 = zeros(1,numFrames);
Y_time1_4 = zeros(1,numFrames);

% initial max
x = 401;
y = 317;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
   
    % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+50:end) = 0;
    % y filter
    GrayFrame(1:y-20,:) = 0;
    GrayFrame(y+50:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time1_4(j) = x;
    Y_time1_4(j) = y;

%     figure(3);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time1_4(j), Y_time1_4(j), 'bo', 'markersize', 10);
%     drawnow
end

S4(1,:) = X_time1_4;
S4(2,:) = Y_time1_4;




%%
%                                     cam2_4                         4

% close all
load('cam2_4.mat')
% implay(vidFrames2_4)

GrayFrameAll = (vidFrames2_4(:,:,:,8:8+384));
numFrames = size(GrayFrameAll,4);
X_time2_4 = zeros(1,numFrames);
Y_time2_4 = zeros(1,numFrames);

% initial max
x = 280;
y = 272;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
   
    % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+50:end) = 0;
    % y filter
    GrayFrame(1:y-20,:) = 0;
    GrayFrame(y+50:end,:) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time2_4(j) = x;
    Y_time2_4(j) = y;

%     figure(3);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time2_4(j), Y_time2_4(j), 'bo', 'markersize', 10);
%     drawnow
end

S4(3,:) = X_time2_4;
S4(4,:) = Y_time2_4;





%%
%                                     cam3_4                         4

% close all
load('cam3_4.mat')
% implay(vidFrames3_4)

GrayFrameAll = (vidFrames3_4(:,:,:,1:1+384));
numFrames = size(GrayFrameAll,4);
X_time3_4 = zeros(1,numFrames);
Y_time3_4 = zeros(1,numFrames);

% initial max
x = 414;
y = 203;

for j = 1:numFrames
    GrayFrame = rgb2gray(GrayFrameAll(:,:,:,j));
   
        % y filter
    GrayFrame(1:y-50,:) = 0;
    GrayFrame(y+50:end,:) = 0;
        % x filter
    GrayFrame(:,1:x-50) = 0;
    GrayFrame(:,x+20:end) = 0;
    
    [Max, index] = max(GrayFrame(:));
    [y,x] = ind2sub(size(GrayFrame), index);
    X_time3_4(j) = x;
    Y_time3_4(j) = y;

%     figure(3);
%     imshow(GrayFrame); 
%     hold on
%     plot(X_time3_4(j), Y_time3_4(j), 'bo', 'markersize', 10);
%     drawnow
end

S4(5,:) = X_time3_4;
S4(6,:) = Y_time3_4;







%%                                     PCA

%%    
%                                     Trial 1

    % SVD
A = S1;
mean_val = mean(A,2);
[m,n] = size(A);
A = A - repmat(mean_val, 1, n);
[U,S,V] = svd(A/sqrt(n-1),'econ');


    % Graphs
figure(1);
subplot(2,2,1)
plot(1:n,S1);
set(gca,'Fontsize',13);
title('Trial 1: Original Data');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(12)
subplot(2,2,2)
plot(1:n, V(:,1:2))
set(gca,'Fontsize',13);
title('Trial 1: Time Series');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(13);
subplot(2,2,3)
plot((diag(S).^2/sum(diag(S).^2)*100), 'ob')
set(gca,'Fontsize',13)
title('Trial 1: Single Values')
xlabel('Index of Single Value')
ylabel('Energy of Single Value')

% figure(14);
subplot(2,2,4)
plot(A'*U(:,1:2))
set(gca,'Fontsize',13)
title('Trial 1: Projected Data')
xlabel('Frame Number')
ylabel('Position (pixel in frame)')





%%    
%                                     Trial 2

    % SVD
A = S2;
mean_val = mean(A,2);
[m,n] = size(A);
A = A - repmat(mean_val, 1, n);
[U,S,V] = svd(A/sqrt(n-1),'econ');


    % Graphs
figure(2);
subplot(2,2,1)
plot(1:n,A);
set(gca,'Fontsize',13);
title('Trial 2: Original Data');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(22)
subplot(2,2,2)
plot(V(:, 1:2));
set(gca,'Fontsize',13);
title('Trial 2: Time Series');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(23);
subplot(2,2,3)
plot((diag(S).^2/sum(diag(S).^2)*100), 'ob')
set(gca,'Fontsize',13)
title('Trial 2: Single Values')
xlabel('Index of Single Value')
ylabel('Energy of Single Value')

% figure(24);
subplot(2,2,4)
plot(A'*U(:,1:2))
set(gca,'Fontsize',13)
title('Trial 2: Projected Data')
xlabel('Frame Number')
ylabel('Position (pixel in frame)')




%%    
%                                     Trial 3

    % SVD
A = S3;
mean_val = mean(A,2);
[m,n] = size(A);
A = A - repmat(mean_val, 1, n);
[U,S,V] = svd(A/sqrt(n-1),'econ');


    % Graphs
figure(3);
subplot(2,2,1)
plot(1:n,A);
set(gca,'Fontsize',13);
title('Trial 3: Original Data');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(32)
subplot(2,2,2)
plot(V(:, 1:3));
set(gca,'Fontsize',13);
title('Trial 3: Time Series');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(33);
subplot(2,2,3)
plot((diag(S).^2/sum(diag(S).^2)*100), 'ob')
set(gca,'Fontsize',13)
title('Trial 3: Single Values')
xlabel('Index of Single Value')
ylabel('Energy of Single Value')

% figure(34);
subplot(2,2,4)
plot(A'*U(:,1:3))
set(gca,'Fontsize',13)
title('Trial 3: Projected Data')
xlabel('Frame Number')
ylabel('Position (pixel in frame)')



%%    
%                                     Trial 4

    % SVD
A = S4;
mean_val = mean(A,2);
[m,n] = size(A);
A = A - repmat(mean_val, 1, n);
[U,S,V] = svd(A/sqrt(n-1),'econ');


    % Graphs
figure(4);
subplot(2,2,1)
plot(1:n,A);
set(gca,'Fontsize',13);
title('Trial 4: Original Data');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(42)
subplot(2,2,2)
plot(V(:, 1:2));
set(gca,'Fontsize',13);
title('Trial 4: Time Series');
xlabel('Frame Number');
ylabel('Position (pixel in frame)');

% figure(43);
subplot(2,2,3)
plot((diag(S).^2/sum(diag(S).^2)*100), 'ob')
set(gca,'Fontsize',13)
title('Trial 4: Single Values')
xlabel('Index of Single Value')
ylabel('Energy of Single Value')

% figure(44);
subplot(2,2,4)
plot(A'*U(:,1:2))
set(gca,'Fontsize',13)
title('Trial 4: Projected Data')
xlabel('Frame Number')
ylabel('Position (pixel in frame)')


















