clear all
close all
clc

%% Importazione immagine e segmenti con conversione
image = medicalImage("000009.dcm");
segmentation_sx = nrrdread("Segmentation_sx.nrrd");
segmentation_dx = nrrdread("Segmentation_dx.nrrd");
image_gray = mat2gray(image.Pixels);
segmentation_sx_gray = mat2gray (segmentation_sx);
segmentation_dx_gray = mat2gray (segmentation_dx);

%% Plot dell'immagine e dei segmenti
figure;

subplot(3, 1, 1);
imshow(image.Pixels, []);
title('Immagine DICOM');

subplot(3, 1, 2);
imshow(segmentation_sx_gray, []);
title('Segmentazione polmone sinistro');

subplot(3, 1, 3);
imshow(segmentation_dx_gray, []);
title('Segmentazione polmone destro');

%% Rotazione segmenti
segmentation_sx_gray_ruotata=permute(segmentation_sx_gray,[2 1 3]); %rotazione scambiando X e Y
segmentation_dx_gray_ruotata=permute(segmentation_dx_gray,[2 1 3]); %rotazione scambiando X e Y

%% Plot dell'immagine e dei segmenti ruotati
figure;
subplot(3, 1, 1);
imshow(image.Pixels, []);
title('Immagine DICOM');

subplot(3, 1, 2);
imshow(segmentation_sx_gray_ruotata, []);
title('Segmentazione polmone sinistro');

subplot(3, 1, 3);
imshow(segmentation_dx_gray_ruotata, []);
title('Segmentazione polmone destro');

%% Trasformazione segmenti in booleani per istogramma
segmentation_sx_bool = segmentation_sx_gray_ruotata > 0.5;
segmentation_dx_bool = segmentation_dx_gray_ruotata > 0.5;

%% Plot istogrammi dei segmenti 
figure;
subplot(2,1,1)
histogram(image_gray(segmentation_sx_bool),256)
title ('Istogramma dei livelli di grigio nella ROI polmone sx')
xlabel('Intensità'), ylabel('Occorrenze');
subplot(2,1,2)
histogram(image_gray(segmentation_dx_bool),256)
title ('Istogramma dei livelli di grigio nella ROI polmone dx')
xlabel('Intensità'), ylabel('Occorrenze');

%% parametri per estrazione feature radiomiche
minIntensity = double(min(image.Pixels(:)));
maxIntensity = double(max(image.Pixels(:)));

%% Estrazione feature radiomiche segmento sinistro
R_sx = radiomics(image.Pixels, segmentation_sx_bool, ...
    DiscreteMethod="FixedBinSize", ...
    DiscreteBinSizeOrBinNumber=25, ...
    Resample=false, ...
    ResampledVoxelSpacing=1, ...
    DataResampleMethod="linear", ...
    MaskResampleMethod="nearest", ...
    ExcludeOutliers=false, ...
    ResegmentationRange=[minIntensity, maxIntensity]); %setting dei parametri per farli combaciare con PyRadiomics

features_sx = selectFeatures(R_sx, ["IntensityEnergy2D", "DiscretisedIntensityEntropy2D", "IntensityKurtosis2D", ...
    "MaximumIntensity2D", "MeanIntensity2D", "MinimumIntensity2D", "IntensityRange2D", ...
    "IntensitySkewness2D", "DiscretisedIntensityUniformity2D", "IntensityVariance2D", ...
    "ContrastAveraged2D", "InverseDifferenceMomentAveraged2D", ...
    "SurfaceAreaMesh2D"]);
area_sx = features_sx.SurfaceAreaMesh2D * image.PixelSpacing(1) * image.PixelSpacing(2); %matlab calcola l'area in pixel^2 l'ho dovuta convertire in mm^2

%% Esportazione feature radiomiche segmento sinistro
feature_names_sx = features_sx.Properties.VariableNames;
values_sx = zeros(length(feature_names_sx),1);
for i=1:length(feature_names_sx) %estrae i valori da ogni colonna
    values_sx(i) = features_sx.(feature_names_sx{i});
end
results_sx = table(feature_names_sx', values_sx, 'VariableNames', {'FeatureName', 'Value'});
results_sx.Value(end) = area_sx; %sostituisco l'ultimo valore della tabella creata che sarebbe l'area in pixel^2 con il valore calcolato in mm^2
writetable(results_sx, "risultati_radiomici_sx.xlsx"); %salvataggio dati ottenuti in file .xlsx

%% Estrazione feature radiomiche segmento destro
R_dx = radiomics(image.Pixels, segmentation_dx_bool, ...
    DiscreteMethod="FixedBinSize", ...
    DiscreteBinSizeOrBinNumber=25, ...
    Resample=false, ...
    ResampledVoxelSpacing=1, ...
    DataResampleMethod="linear", ...
    MaskResampleMethod="nearest", ...
    ExcludeOutliers=false, ...
    ResegmentationRange=[minIntensity, maxIntensity]);
features_dx = selectFeatures(R_dx, ["IntensityEnergy2D", "DiscretisedIntensityEntropy2D", "IntensityKurtosis2D", ...
    "MaximumIntensity2D", "MeanIntensity2D", "MinimumIntensity2D", "IntensityRange2D", ...
    "IntensitySkewness2D", "DiscretisedIntensityUniformity2D", "IntensityVariance2D", ...
    "ContrastAveraged2D", "InverseDifferenceMomentAveraged2D", ...
    "SurfaceAreaMesh2D"]);
area_dx = features_dx.SurfaceAreaMesh2D * image.PixelSpacing(1) * image.PixelSpacing(2);

%% Esportazione feature radiomiche segmento destro
feature_names_dx = features_dx.Properties.VariableNames;
values_dx = zeros(length(feature_names_dx),1);
for i=1:length(feature_names_dx)
    values_dx(i) = features_dx.(feature_names_dx{i});
end
results_dx = table(feature_names_dx', values_dx, 'VariableNames', {'FeatureName', 'Value'});
results_dx.Value(end) = area_dx;
writetable(results_dx, "risultati_radiomici_dx.xlsx"); %salvataggio dati ottenuti in file .xlsx