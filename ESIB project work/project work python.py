#Importazione librerie utili durante il codice
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label,find_contours
from scipy.stats import skew, kurtosis, entropy
import SimpleITK as sitk
from radiomics import featureextractor
import csv #libreria per estrarre i risulatti in tabella

#definizione di una funzione
def mat2gray(img):
    """
    Funzione che esegue la normalizzazione.
    Comoda per essere richiamata dentro il programma
    """
    img= img.astype(np.float32)
    return (img - img.min())/ (img.max()- img.min())


#Importazione file
#Indico a python dove trovare i file, in questo caso sono nella stessa cartella dello script
imagePath="000009.dcm"
segmentation_sxPath="Segmentation_sx.nrrd"
segmentation_dxPath="Segmentation_dx.nrrd"

#Visualizzo il contenuto
image= sitk.ReadImage(imagePath)
segmentation_sx=sitk.ReadImage(segmentation_sxPath)
segmentation_dx=sitk.ReadImage(segmentation_dxPath)

#Conversione immagine e maschere in array per poterla plottare
image_array= sitk.GetArrayFromImage(image)
segmentation_sx_array= sitk.GetArrayFromImage(segmentation_sx)
segmentation_dx_array=sitk.GetArrayFromImage(segmentation_dx)

#Conversione immagini in 2D
image_2d= image_array[0]
segmentation_sx_2d= segmentation_sx_array[0]
segmentation_dx_2d= segmentation_dx_array[0]

#Visualizzazione immagine e segmenti
plt.subplot (3,1,1)
plt.imshow(image_2d, cmap="gray")
plt.title("Immagine DICOM")
plt.axis('off')

plt.subplot(3,1,2)
plt.imshow(segmentation_sx_2d, cmap="gray")
plt.title("Segmentazione polmone sinistro")
plt.axis('off')

plt.subplot(3,1,3)
plt.imshow(segmentation_dx_2d, cmap="gray")
plt.title("Segmentazione polmone destro")
plt.axis('off')
plt.subplots_adjust(hspace=0.5) #hspace distanzia i grafici verticalmente
plt.show()

#Normalizzazione immagine
image_norm=mat2gray(image_2d)

#Plot di immagine, immagine normalizzata e segmenti
plt.subplot (4,1,1)
plt.imshow(image_2d, cmap="gray")
plt.title("Immagine DICOM")
plt.axis('off')

plt.subplot(4,1,2)
plt.imshow(image_norm, cmap="gray")
plt.title("Immagine normalizzata")
plt.axis('off')

plt.subplot(4,1,3)
plt.imshow(segmentation_sx_2d, cmap="gray")
plt.title("Segmentazione polmone sx")
plt.axis('off')

plt.subplot(4,1,4)
plt.imshow(segmentation_dx_2d, cmap="gray")
plt.title("Segmentazione polmone dx")
plt.axis('off')
plt.subplots_adjust(hspace=0.5)

plt.show()

#Estrazione istogramma dei livelli di grigio
roi_pixels_sx= image_norm[segmentation_sx_2d==1]
roi_pixels_dx= image_norm[segmentation_dx_2d==1]

#Plot istogramma
plt.subplot (2,1,1)
plt.hist(roi_pixels_sx, bins=256)
plt.title("Istogramma dei livelli di grigio nella ROI polmone sx")
plt.xlabel("Intensità")
plt.ylabel("Occorrenze")

plt.subplot (2,1,2)
plt.hist(roi_pixels_dx, bins=256)
plt.title("Istogramma dei livelli di grigio nella ROI polmone dx")
plt.xlabel("Intensità")
plt.ylabel("Occorrenze")

plt.subplots_adjust(hspace=1) 
plt.show()

#Impostazione manuale del pixel spacing
image.SetSpacing((0.139,0.139,1.0))
#Imposto manualmnete il pixel spacing dopo averlo letto da 3d slicer in quanto pyton lo importa male e ciò da errore quando usa pyradiomics perche dice che l'immagine e la maschera non hanno la stessa dimensione


#Estrazione delle feature

#Definizione dell'estrattore
fex = featureextractor.RadiomicsFeatureExtractor()

#Selezioniamo le feature da estrarre
fex.disableAllFeatures() #disabilito tutte le feature
fex.enableFeaturesByName(shape=['SurfaceArea']) #Estrazione di SurfaceArea dalle shape feature
fex.enableFeaturesByName(glcm=['Contrast', 'Idm']) #Estrazione di Contrast e Idm dalle glcm feature
fex.enableFeaturesByName(firstorder=['Energy','Entropy','Kurtosis','Maximum','Mean','Minimum','Range','Skewness','Uniformity','Variance']) #Estrazione delle feature riportate dalle firstorder feature

#Estrazione vera e propria delle feature
result_sx = fex.execute(image, segmentation_sx)
result_dx= fex.execute(image, segmentation_dx)

#Salvataggio dati ottenuti in file .csv
with open("risultati_radiomici_sx.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Intestazioni (prima riga)
    writer.writerow(["Feature", "Valore"])
    
    # Ogni feature in una riga
    for key, value in result_sx.items():
        writer.writerow([key, value])

with open("risultati_radiomici_dx.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Intestazioni (prima riga)
    writer.writerow(["Feature", "Valore"])
    
    # Ogni feature in una riga
    for key, value in result_dx.items():
        writer.writerow([key, value])
