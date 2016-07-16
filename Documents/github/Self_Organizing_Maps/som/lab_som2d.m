function [som, grid, qe, te] = lab_som2d (trainingData, neuronCountW, neuronCountH, trainingSteps, startLearningRate, startRadius)
    [M,N] = size(trainingData);
    % startLearningRate should be equal to 0.1 (heuristic)
    % startRadius should be equal to max(latticeHeight,latticeWidth)/2
    
    T1 = 1000; % learning rate constant
    T2 = 1000/log(startRadius); % radius constant
    qe = 0; % quantization error
    te = 0; % topographic error
    
    % 1. Initilise weight to samll, random values
    neuronCount = neuronCountW*neuronCountH;
    neurons = rand(neuronCount,N);
    flag = 1;
    % Initialising grid according to requirement
    %     for i = 1:neuronCountW % column
    %         flag = i;
    %         for j = 1:neuronCountH % row
    %             grid(flag,:) = [j i];
    %             flag = flag + neuronCountW;
    %         end
    %     end

    % Initialising grid according to requirement
    for i = 1:neuronCountH % row
        for j = 1:neuronCountW % column
            grid(flag,:) = [i j];
            flag = flag + 1;
        end
    end
     
    % 2. Repeat until covergence
    for t = 1:trainingSteps    
        % 2a. Select input pattern - x
        x = trainingData(randi(M,1,1),:); 
        % 2a1. Find winner
        minScore = eucdist(x,neurons(1,:));
        winner = 1;
        minScore2 = eucdist(x,neurons(1,:));
        winner2 = 1;
        for n = 2:neuronCount
            score = eucdist(x,neurons(n,:));
            if score<minScore
                minScore = score;
                winner = n;
            end
        end
        qe = qe + eucdist(x,neurons(winner,:));
        % find second best matching unit for topographical error
        for n = 2:neuronCount
            score = eucdist(x,neurons(n,:));
            if score<minScore & n~=winner
                minScore2 = score;
                winner2 = n;
            end
        end
        te = te + (norm(grid(winner,:)-grid(winner2,:),1)~=1);
        % 2a2. Update winner & its all neighbours
        lcRule = startLearningRate * exp(-t/T1);
        radiusRule = startRadius * exp(-t/T2);
        for n = 1:neuronCount
            factor = exp(-(norm(grid(winner,:)-grid(n,:),1))^2/(2*radiusRule^2));
            %factor = exp(-(sum(abs(grid(winner,:)-grid(n,:)),2))^2/(2*radiusRule^2));
            neurons(n,:) = neurons(n,:)+lcRule*factor*(x-neurons(n,:));
        end
        
         % 2b. Decrease learning rate by t = t + 1
         % 2c. Decrease neighbourhood size by t = t + 1
     if(mod(t,500)==0)
        lab_vis2d(neurons,grid,trainingData);
        drawnow;
     end
     
    end
    som = neurons;
    qe = qe/trainingSteps;
    te = te/trainingSteps;
end

% data = nicering;
% [som, grid] = lab_som2d(data, 15, 3, 5000, 1, 10);
% lab_vis2d(som, grid, data);


% som = lab_som2d (trainingData, neuronCountW, neuronCountH, trainingSteps, startLearningRate, startRadius)
% -- Purpose: Trains a 2D SOM, which consists of a grid of
%             (neuronCountH * neuronCountW) neurons.
%             
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <grid> returns the location of the neurons in the grid
% -- <neuronCountW> number of neurons along width
% -- <neuronCountH> number of neurons along height
% -- <trainingSteps> number of training steps 
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius used to specify the initial neighbourhood size
%

% TODO:
% The student will need to copy their code from lab_som() and
% update it so that it uses a 2D grid of neurons, rather than a 
% 1D line of neurons.
% 
% Your function will still return the a weight matrix 'som' with
% the same format as described in lab_som().
%
% However, it will additionally return a vector 'grid' that will
% state where each neuron is located in the 2D SOM grid. 
% 
% grid(n, :) contains the grid location of neuron 'n'
%
% For example, if grid = [[1,1];[1,2];[2,1];[2,2]] then:
% 
%   - som(1,:) are the weights for the neuron at position x=1,y=1 in the grid
%   - som(2,:) are the weights for the neuron at position x=2,y=1 in the grid
%   - som(3,:) are the weights for the neuron at position x=1,y=2 in the grid 
%   - som(4,:) are the weights for the neuron at position x=2,y=2 in the grid
%
% It is important to return the grid in the correct format so that
% lab_vis2d() can render the SOM correctly
