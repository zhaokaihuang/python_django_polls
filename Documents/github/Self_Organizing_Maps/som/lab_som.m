function [som, qe, te] = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
    [M,N] = size(trainingData);
    T1 = 1000; % learning rate constant
    T2 = 1000/log(startRadius); % radius constant
    qe = 0; % quantization error
    te = 0; % topographic error
    
    % 1. Initilise weight to some samll, random values
    neurons = rand(neuronCount,N);
    
    % 2. Repeat until covergence
    for t = 1:trainingSteps   
        % 2a. Select input pattern - x
        x = trainingData(randi(M,1,1),:);       
        % 2a1. Find winner - BMU
        minScore = eucdist(x,neurons(1,:));     
        winner = 1;
        winner2 = 1;
        minScore2 = eucdist(x,neurons(1,:));
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
        te = te + (abs(winner-winner2)~=1);
        % 2a2. Update winner & its all neighbours
        lcRule = startLearningRate * exp(-t/T1);
        radiusRule = startRadius * exp(-t/T2);
        for n = 1:neuronCount
           factor = exp(-abs(winner-n)^2/(2*radiusRule^2));
           neurons(n,:) = neurons(n,:)+lcRule*factor*(x-neurons(n,:));
        end
     % 2b. Decrease learning rate by t
     % 2c. Decrease neighbourhood size by t
    %disp(t)
     if(mod(t,500)==0)
        lab_vis(neurons,trainingData);
        drawnow;
     end
     
    end
    som = neurons;
    qe = qe/trainingSteps;
    te = te/trainingSteps;
end

% data=nicering;
% [som, qe, te]=lab_som(data, 20, 5000, 0.1, 20);
% lab_vis(som, data);


% som = lab_som (trainingData, neuronCount, trainingSteps, startLearningRate, startRadius)
% -- Purpose: Trains a 1D SOM i.e. A SOM where the neurons are arranged
%             in a single line. 
%             
% -- <trainingData> data to train the SOM with
% -- <som> returns the neuron weights after training
% -- <neuronCount> number of neurons 
% -- <trainingSteps> number of training steps 
% -- <startLearningRate> initial learning rate
% -- <startRadius> initial radius used to specify the initial neighbourhood size

% TODO:
% The student will need to complete this function so that it returns
% a matrix 'som' containing the weights of the trained SOM.
% The weight matrix should be arranged as follows, where
% N is the number of features and M is the number of neurons:
%
% Neuron1_Weight1 Neuron1_Weight2 ... Neuron1_WeightN
% Neuron2_Weight1 Neuron2_Weight2 ... Neuron2_WeightN
% ...
% NeuronM_Weight1 NeuronM_Weight2 ... NeuronM_WeightN
%
% It is important that this format is maintained as it is what
% lab_vis(...) expects.
%
% Some points that you need to consider are:
%   - How should you randomise the weight matrix at the start?
%   - How do you decay both the learning rate and radius over time?
%   - How does updating the weights of a neuron effect those nearby?
%   - How do you calculate the distance of two neurons when they are
%     arranged on a single line?








