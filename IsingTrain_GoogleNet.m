pathToImages = fullfile("C:\\Users\\swarn\\Desktop\\Apply Academic\\", "ML_Images") ; %% Change the path accoring to you
%Create a  Image Datastore
image_Datastore = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames') ;

disp(image_Datastore.Labels)
no_EachLabel = countEachLabel(image_Datastore) ; % count no of labels in each category
%disp(no_EachLabel)

% Create Train & Test Image Variables
[trainingSet, testSet] = splitEachLabel(image_Datastore, 0.8, 'randomize') ;

% Call the pretrained network
net = googlenet ;
disp(net.Layers)
first_Layer_InputSize = net.Layers(1).InputSize ;
%disp(net.Layers(1).InputSize)

disp(isa(net,'SeriesNetwork'))

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end

disp(lgraph)

%names of the two layers to replace
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
disp([learnableLayer,classLayer])

% Number of categories of Temperature
numClasses = numel(categories(trainingSet.Labels));
disp(numClasses)

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%Freeze the top 10 Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%{
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%}

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

% Resize the training Data
augimdsTrain = augmentedImageDatastore(first_Layer_InputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);

% Resize the test Data
augimdsTest = augmentedImageDatastore(first_Layer_InputSize(1:2),testSet);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsTest, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train GoogleNet
net = trainNetwork(augimdsTrain,lgraph,options);

% save the modified GoogleNet
Ising_GoogleNet = net ;
save("D:\TU\Classifier\Ising_GoogleNet")  %% Change the saving path
