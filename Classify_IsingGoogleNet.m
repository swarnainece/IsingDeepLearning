pathToImages = fullfile("C:\\Users\\swarn\\Desktop\\Apply Academic\\", "ML_Images") ;

%Create a  Image Datastore
image_Datastore = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames') ;

%disp(image_Datastore.Labels)
no_EachLabel = countEachLabel(image_Datastore) ; % count no of labels in each category
disp(no_EachLabel)

% Create Train & Test Image Variables
[trainingSet, testSet] = splitEachLabel(image_Datastore, 0.7, 'randomize') ;

%Load teh pretrained Classifier.
load ("D:\TU\Classifier\Ising_GoogleNet") ;

%analyzeNetwork(Ising_GoogleNet)

first_Layer_InputSize = Ising_GoogleNet.Layers(1).InputSize ;

% Resize the test Data % first two index values of the Layers(1).InputSize array
augimdsTest = augmentedImageDatastore(first_Layer_InputSize(1:2),testSet); 

%Classify Validation Images
[YPred,probs] = classify(Ising_GoogleNet,augimdsTest);
accuracy = mean(YPred == testSet.Labels) ;

% Calculating the confusion chart
%confusionchart(testSet.Labels,YPred)


% Pick random 20 images
idx = randperm(numel(testSet.Files), 1);

disp(accuracy)

no_of_Labels_testImgs = testSet.Labels ;
%disp(no_of_Labels_testImgs)

%{
label = classify(Ising_GoogleNet,I);
figure
imshow(I)
title(string(label))

%}

for i = 1:1
    subplot(1,1,i)
    I = readimage(testSet,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
