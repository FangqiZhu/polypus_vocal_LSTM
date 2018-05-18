% ================================================================= %
% speech_RNN.m (verion 2)
% Description: Speech data preprocessing using recurrent neural network
% Datasets: two types (abnormal, normal)
%           abnormal ([vowel "a:" 1]),  size: 14700400 row vector
%           normal ([vowel "a:" 1]),    size: 13844300 row vector
%           abnormal ([vowel "i:" 1]),  size: 14700400 row vector
%           normal ([vowel "i:" 1]),    size: 13844300 row vector
% cleaning data memo: 'Data/abnormal/11txt/11a1.txt' the file contains 
%                      half of total missing data, drop the column
% Recurrent neural network Configuration: BiLSTM + Adam Optimizer
% Author: Fangqi Zhu, Research Assistant
% Affiliation: University of Texas at Arlington
% Last modified: 04/21/2018
% ================================================================= %

clear all; close all; clc;

%% dataloading
sp_normal_a2 = load('normal_a2.mat');                      
sp_abnormal_a2 = load('abnormal_a2.mat');  
sp_normal_a2 = sp_normal_a2.AA;
sp_abnormal_a2 = sp_abnormal_a2.AA;

%% data summary and visualization
fprintf("Normal Data summary: ")
fprintf('\n');
summary_sp(sp_normal_a2(:,1));

fprintf("Abnormal Data summary: ")
fprintf('\n');
summary_sp(sp_abnormal_a2(:,1));

% figure(1)
% subplot(211);
% plot(1:length(sp_normal_a2), sp_normal_a2(:,5), 'k');
% subplot(212);
% plot(1:length(sp_abnormal_a2), sp_abnormal_a2(:,5), 'k');

%% Frequency Analysis 

% Parameter setting
Fs = 192000;                % sampling frequency
nfft = 2048;                % number of FFT
noverlap = [];              % number of opverlapped samples (50% default)

% Short-Time Fourier Transform Analysis
figure(1);
spectrogram(sp_normal_a2, rectwin(nfft), noverlap, nfft, Fs, 'yaxis');  

figure(2);
spectrogram(sp_abnormal_a2, rectwin(nfft), noverlap, nfft, Fs, 'yaxis');


[spec_normal_i1, ~, ~] = spectrogram(sp_normal_a2, rectwin(nfft),...
                                  noverlap, nfft, Fs); 


[spec_abnormal_i1, ~, ~] = spectrogram(sp_abnormal_a2, rectwin(nfft),...
                                  noverlap, nfft, Fs);


t_normal = 0:1/Fs: (length(sp_normal_a2) - 1)/Fs;    % time index
f = (-nfft/2 :nfft/2 - 1)*Fs/nfft;                   % frequency index (Hz)

% normal case: vocal signal in time and frequency domain

normal_freq = fftshift(fft(sp_normal_a2, nfft));
normal_freq = normal_freq(1:nfft);
normal_freq = abs(normal_freq);

figure(3);
subplot(211);
plot(t_normal, sp_normal_a2);
xlabel('Time (ms)'); ylabel('Amplitude');
subplot(212);
stem(f, normal_freq); xlabel('Frequency (Hz)'); ylabel('Amplitude');

% abnormal case: vocal signal in time and frequency domain
t_abnormal = 0:1/Fs: (length(sp_abnormal_a2) - 1)/Fs;

abnormal_freq = fftshift(fft(sp_abnormal_a2, nfft));
abnormal_freq = abnormal_freq(1:nfft);
abnormal_freq = abs(abnormal_freq);

figure(4);
subplot(211);
plot(t_abnormal, sp_abnormal_a2);
xlabel('Time (ms)'); ylabel('Amplitude');
subplot(212);
stem(f, abnormal_freq); xlabel('Frequency (Hz)'); ylabel('Amplitude');

%% Audio play
% play the fifth sequence sound
% Fs = 1500;                               % sample rate: 44100 Hz
% [sound_normal, Fs] = audioread(sp_normal_a2); % just for debug, don't use it
% [sound_abnormal, Fs] = audioread(sp_abnormal_a2);
% sound(sound_normal, Fs);
% sound(sound_abnormal, Fs);

%% Data fit generator
[r, c] = size(spec_normal_a2);

X_sp_normal = abs(spec_normal_a2(:, 1:13500)');         % Dim: r*27000
X_sp_normal = X_sp_normal/max(max(X_sp_normal));

X_sp_abnormal = abs(spec_abnormal_a2(:, 1:13500)');     % Dim: r*27000
X_sp_abnormal = X_sp_abnormal/max(max(X_sp_abnormal));

Y_sp_normal = 1 * ones(r, 1);
Y_sp_abnormal = -1 * ones(r, 1);

%% Leave One Out Cross-Validation

X_Train = [X_sp_normal(:, 1: floor(r*0.8)) X_sp_abnormal(:, 1:floor(r*0.8))];  
X_Train = num2cell(X_Train, 1);  % change the data format to cell 

X_Test = [X_sp_normal(:, (floor(r*0.8)+1):r)  X_sp_abnormal(:, (floor(r*0.8)+1):r)]; 
X_Test = num2cell(X_Test, 1);    % change the data format to cell

Y_Train = [Y_sp_normal(1: floor(r*0.8), :); Y_sp_abnormal(1: floor(r*0.8), :)];
Y_Train = categorical(Y_Train);  % change the response to categorical data

Y_Test = [Y_sp_normal((floor(r*0.8)+1):r, 1); Y_sp_abnormal((floor(r*0.8)+1):r, 1)];
Y_Test = categorical(Y_Test);    % change the response to categorical data


%% LSTM Network Architecture
inputSize = 13500;
outputSize = r;
outputMode = 'last';
numClasses = 2;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(outputSize,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 20;
miniBatchSize = 500;
shuffle = 'once';

options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ...
    'Verbose',0, ...
    'MaxEpochs',100,...
    'Plots','training-progress');

%% Train LSTM Network
net = trainNetwork(X_Train,Y_Train,layers,options);

%% Test LSTM Network
Y_Pred = classify(net,X_Test,'MiniBatchSize',miniBatchSize);
acc = sum(Y_Pred == Y_Test)./numel(Y_Test);
fprintf('The accuracy of the Test data: %.f %%', acc* 100);
fprintf('\n');





































% sp_normal_a1 = csvread('sp_normal_a1.csv',...
%                      1, 0,[1 0 1e6 9]);   % normal data, size: 1e6 * 10   
% sp_abnormal_a1 = csvread('sp_abnormal_a1.csv',...
%                      1, 0,[1 0 1e6 9]);   % abnormal data, size: 1e6 * 10


% X_sp_normal = reshape(X_sp_normal, 513*10, 2700);
% X_sp_normal = X_sp_normal';
% X_sp_abnormal = reshape(X_sp_abnormal, 513*10, 2700);
% X_sp_abnormal =X_sp_abnormal';

% label setting: 513 labels for each case




% Data_normal = [X_sp_normal' Y_sp_normal];
% Data_abnormal = [X_sp_abnormal' Y_sp_abnormal];
% X_Train = reshape(X_Train, [10000, 1, 1600]);          % 10000-by-1600
% X_Test = reshape(X_Test, [10000, 1, 400]);             % 10000-by-400

% X_sp_normal = reshape(sp_normal_a2_norm, 10000, 1380);
% X_sp_abnormal = reshape(sp_normal_a2_norm, 10000, 1380);
% % label setting: 1000 labels
% Y_sp_normal = 1*ones(1380, 1);
% Y_sp_abnormal = -1*ones(1380, 1);
% 
% % Data_normal = [X_sp_normal' Y_sp_normal];
% % Data_abnormal = [X_sp_abnormal' Y_sp_abnormal];
% X_Train = [X_sp_normal(:, 1:1000) X_sp_abnormal(:, 1:1000)];  % 10000-by-1600
% %X_Train = reshape(X_Train, [10000, 1, 1600]);
% X_Train = num2cell(X_Train, 1);  % change the data format to cell 
% 
% X_Test = [X_sp_normal(:, 1001:1380)  X_sp_abnormal(:, 1001:1380)]; % 10000-by-400
% %X_Test = reshape(X_Test, [10000, 1, 400]);
% X_Test = num2cell(X_Test, 1);    % change the data format to cell
% 
% Y_Train = [Y_sp_normal(1:1000, :); Y_sp_abnormal(1:1000, :)];
% Y_Train = categorical(Y_Train);  % change the response to categorical data
% 
% Y_Test = [Y_sp_normal(1001:1380, 1); Y_sp_abnormal(1001:1380, 1)];
% Y_Test = categorical(Y_Test);    % change the response to categorical data




