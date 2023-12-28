# Encoder-Decoder Chess Disease Identification Model

## Abstract
This project focuses on developing an Encoder-Decoder-based model for identifying diseases in chest X-ray radiological images. The purpose is to reduce the workload of radiologists and provide accurate disease identification and captions for patient understanding. The research is conducted on the COVID Radiographic Dataset from Kaggle, aiming to automate disease identification and enhance the diagnostic process.

## Introduction
Radiological images, especially chest X-rays, play a crucial role in diagnosing various diseases. This research aims to leverage deep learning models to automate disease identification and generate descriptive captions for chest X-ray images. By doing so, the workload of radiologists can be reduced, leading to quicker and more efficient diagnoses. The Encoder-Decoder model, combined with CNNs, is used for this purpose.

## Method
The method involves training multiple CNN models (Xception, Inception-v3, VGG16, Inception-Resnet-v2, and EfficientNetB7) for feature extraction. The embedded data from CNNs is then passed through either LSTM or Transformer-based encoders to generate context vectors. These context vectors, along with provided captions, are fed into the decoder for disease identification and caption generation. The models are trained on the COVID-19 Radiographic Dataset.

## Dataset Used
The COVID-19 Radiographic Dataset, winner of the Kaggle COVID-19 Dataset Award, is utilized. It includes images categorized into COVID-19 positive, Lung Opacity, Viral Pneumonia, and Normal Lungs. Captions for images are manually provided for training.

## CNN Models
The CNN models used are VGG16, Inception-v3, Inception-Resnet-v2, Xception, and EfficientNetB7. Each model is employed for feature extraction from chest X-ray images.

## LSTM and Transformer
LSTM and Transformer architectures are used for encoding the embedded data from CNN models. LSTM is a Recurrent Neural Network architecture, while Transformers use self-attention mechanisms for processing entire sequences at once.

## Evaluation and Results
The models are evaluated based on accuracy, validation loss, BLEU, RougeL, etc. Results show that the Xception-Transformer model with batch normalization and max pooling provides the least validation loss, while the Inception-Resnet-v2-LSTM model with batch normalization and max pooling achieves the highest accuracy.

## Conclusion
This research compares multiple Encoder-Decoder models for disease identification in chest X-ray images. The Inception-Resnet-v2-LSTM model performs well in terms of accuracy, while the Xception-Transformer model excels in minimizing validation loss. Further improvements can be made by exploring advanced methods such as object detection models and incorporating larger datasets.

