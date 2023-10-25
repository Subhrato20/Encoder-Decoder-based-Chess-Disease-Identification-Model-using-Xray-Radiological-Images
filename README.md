# Encoder-Decoder-based-Chess-Disease-Identification-Model-using-Xray-Radiological-Images

Encoder Decoder based Chess Disease Identification Model using Xray Radiological Images

Abstract
Purpose: Generating Captions Label and Identifying objects for an image is one of the state-of-the-art problem statements. It can be used in the field of Radiology as it can reduce the need for manual analysis by radiologists and provide patients with a human like experience. In recent time we had already seen increase in level of work-related stress among doctors and radiologists, automated caption generation can help with that also.
Method: In this research, we have used an Encoder-Decoder based model for identifying disease with multiple CNN models for feature extraction of the image data then encoder is used to the generate context vectors and then the decoder generates captions from the context vector. After training all the models we have compared them. In this research, we have used radiological dataset for training, we have used COVID Radiographic Dataset from Kaggle.
Result: The first result which we found is that Xception based model produces better validation loss, Validation loss is a metric used to evaluate the generalization ability of a machine learning model and the second result is that Inception-resnet-v2 based models provided better accuracy, with Inception-Resnet-v2-LSTM model producing best max accuracy.
Conclusion: This study provided a comparison of multiple image captioning models. Our models are capable of predicting good quality captions for both of the dataset. With more training, it may be possible to provide more better outcome with respect to Chest X-ray dataset, improving the quantity of the image in each class can provide more better outcomes.
Keywords: Radiology, Image Captioning, Convolutional Neural Networks (CNN), Encoder-Decoder.

Introduction
Radiological images like Chest X-ray images can be used and are used for detecting a lot of diseases including diseases like COVID-19, Pneumonia, Lung Opacity, and a lot more other disease. Yao, L. [28] classified more diseases with chest X-ray images like Atelectasis, Cardiomegaly, Pneumothorax and etc. Traditionally it has been the job of a Radiologist to do the diagnosis of the Xray Images and predict the abnormalities but with the advancement in the technology and the emergence of deep learning models, some of the workload of these professionals can be reduced. As we are able to design models that can identify disease and then generate a text-based description of the provided radiological images. These models can use neural networks to learn from the large-scale dataset which can help them to further enhance the process of identifying the cause and generate the description. 
According to a study done by Elshami et.al [29] During the time when COVID-19 was at its peak, the workload of Doctors as well as radiologists increased which caused work-related stress to be increased by 42.9% with fear score and anxiety also increased during that period. As said earlier, using a deep learning-based model, which can provide general purpose image captioning and enhance it to help us identify disease and generate descriptions unique to patients, we can reduce the workload of the radiologist [1]  and can save their time too which they can use to provide to some other important things. These models can also help improve the fluidity of the diagnosis as well as it can provide human-like descriptions and can provide patient-friendly experiences as they can provide descriptions or captions for Xray Images that can be understood by anybody. Also with certain amount of high training of the model, we can provide high accuracy results.
Describing Image or Image Captioning is one of the state-of-the-art research that can not only be used in radiological data but also can be multi-domain, like providing autonomous understanding for visually impaired people; it has several other major roles in image-based search in browsers and e-commerce websites. Several recent studies are being done in this field which lured the interest of researchers in the field of Computer Vision and Natural Language Processing. Some studies even have implemented pre-trained language models like GPT [2] and BERT. 
The use of Encoder-Decoder-based model is one of the more successful ones and are used more often in these kinds of research. For this research we have also opted to use encoder decoder based model. In this model we have used, the Convolutional Neural Networks (CNNs) as pre-encoder, which is used to extract image features which then encoder used to generate context to be provided to the decoder, the decoder then uses the power of Natural Language Processing to generate appropriate description for the image. These models with some variations are often used in studies in various fields for generating good-quality captions. Transfer Learning, that entails applying knowledge obtained from one problem to improve performance on a separate but related problem, has also played a vital role in the advancement of these kinds of models as we can now use CNN models like Xception [30], Efficientnet [33] and etc, which has pre-trained weights, trained on ImageNet dataset and can then be used to train on much smaller sized custom dataset as per the required application. We also have leveraged the usefulness of transfer learning in our encoder decoder model as we have used pretrained CNN based image classification model for trining.
Through this paper, we aimed to provide a comparative study of multiple encoder-decoder-based models and to find the better one out of it using the inspired by De Falco, I. et. al. [23] where they classified COVID 19, we have used COVID-19 Radiographic Dataset[22] from Kaggle which is also the winner of the COVID-19 Dataset Award by Kaggle Community. Which consist of classes: COVID-19, Viral Pneumonia, Lung Opacity, and Normal Lungs. In the dataset we have implemented multiple encoder decoder based models and compared their accuracy scores, validation loss. The CNN based image classification models used for feature extraction of the Chest Xray image data used in this research are Xception [30], Inception-V3 [31], VGG16 [32], Inception-Resnet-V2 [34], and EfficientNetB7 [33]. And Long Short Term Memory (LSTM) [35] and Transformers [36] are used for encoding the embedded data from CNN model to context vector and then decoding the following context vectors to identify disease out of those 4 classes and providing unique descriptions for them.

Method
Our method includes a comparative study of multiple encoder-decoder based Image Captioning models. Inspired by the image captioning model by keras team which involves a Convolutional Neural Network (CNN) which extracts features from the image and embeds the image which is then passed to the encoder to generate the context vector then it gets decoded to generate captions. For training the model we have used COVID-19 Radiography Database [22]. After training we evaluated the models.

1.1	Image Captioning on COVID-19 Radiography Database Dataset
We have trained the image captioning model to perform a comparative study. The general structure of the captioning model is described in Fig(1).

 
Fig. 1. Block Diagram of Image Captioning model for Covid Radiographic Dataset.
To do this study we have taken 12,000 sets of sample images paired with their respective disease description from the Covid Radiographic Dataset [22] and divided them using the 80:20 rule, inspired by [7] Puscasiu et. al. As there was no disease description included with the dataset. So, we had provided captions for all the images with respect to the class it belongs to. For training our model in broader vocabulary, we have provided 15 captions for each of the abnormality and divided it using the random function to the model in the image as shown in table 1 for COVID-19 Class.
Table 1. Captions provided for COVID class similarly for Viral Pneumonia, Lung Opacity and Normal Lungs.
Anomaly	Captions
Captions for COVID-19 Class	'Covid is the root cause of this case.',
	'The scenario in question is directly linked to Covid.',
	'The current situation is caused by Covid.',
	'Covid is the driving force behind this matter.',
	'This is an instance where Covid is the main problem.',
'The patient is diagnosed with Covid.',
'The cause of the illness is Covid.',
'Covid is the primary factor in this case.',
'The issue at hand is due to Covid.',
'The diagnosis is Covid-related.',
'Covid is the main concern in this scenario.',
'The symptoms are consistent with Covid.',
'The patient is being treated for Covid.',
'Covid is the main culprit in this case.',
'The situation is complicated by Covid.',


 In this research, we have used multiple Convolutional Neural Networks (CNNs) based imager classification models for extracting the features or we can say embedding the images, which includes Inception-v3 [31], Inception-v3 with Batch Normalizations and Max Pooling, Inception-ResNet-v2 [34], Inception-ResNet-v2 with Batch Normalizations and Max Pooling, VGG16 [32], Xception [30] with Batch Normalizations and Max Pooling and EfficientNetB7 [33] with Batch Normalizations and Max Pooling. Batch Normalizations are generally used to increase training stability and make the learning process quicker. The training process is made more stable and effective by normalizing the activations of the neurons in each stratum of the network and it also prevents the model from overfitting. Also, we have added max pooling to the models which are used to add spatial variance to the model that helps the network to become tolerant to minor rotations or translations in the source picture. The model performs feature extraction as discussed using the CNN from the given training set and helps in embedding the image. Feature extraction is a method that extracts useful and meaningful information from raw data to create a more compact representation, in the embedded form of a feature vector. It entails converting input data into a set of numerical features that capture the fundamental traits or patterns necessary for a certain activity. The embedded data extracted from feature extraction is then then passed to transformers [35] and LSTM [36] based encoders individually that is having a dense sub-layer and uses sequence2sequence learning to generate the context vector for the decoder. The context vector i.e. the output from the encoder along with a sequence of  provided description for an image is then fed to the decoder respective to the encoder i.e., If the encoder is using LSTM architecture then the decoder will also be LSTM and if the encoder is having Transformer architecture then the decoder will also be Transformer based as shown in Fig (2) and Fig (3).
 
Fig. 2 LSTM based Encoder and Decoder.

 
Fig. 3 Transformer based Encoder and Decoder.
 The decoder then takes the input which it then uses to generate captions. The casual attention mask is used to stop the decoder from paying attention to upcoming time steps while it is being trained. The mask is a lower triangular matrix with a (i,j) member that is either 0 or 1 depending on whether j > i. We have trained the model in 2 runs first for 10 epochs to calculate the accuracy and validation loss and for evaluating output in the evaluation matrix i.e. BLEU and RougeL, etc, 5 epochs. In the training model, there is also an early stopping callback with a patience of 3 [37].  Fig (5) shows an example of output generated by one the implemented model.
 
Fig. 5 Example output description generated by Inception-Resnet-v2 and transformer with batch normalization and max pooling.


1.2	Dataset Used
For image captioning in Chest Xray images, we have used COVID-19 Radiography Database [22] which is also the winner of the COVID-19 Dataset Award by Kaggle Community it was developed by researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh and it consists of images under the category of COVID-19 positive, Lung Opacity i.e., Non-COVID Lung Infection, Viral Pneumonia, and Normal Lungs. The diagnoses for the images were then graded by two expert physicians before being cleared for training in the AI system. As the dataset did not have captions for the images, we had created our own caption dataset for all the images present in the following dataset and then used them for training our image captioning models.

1.3	Convolutional Neural Network (CNN) Models
The CNN model used for the study are VGG16 [32], Inception-v3 [31], Inception-Resnet-v2 [34], Xception [30], and EfficientNetB7 [33].

VGG16: The VGG16 architecture was introduced in 2014 and has 16 convolutional layers. To reduce the spatial dimensions, the network employs max pooling layers and 3x3 convolutional filters throughout. Although VGG16 has a straightforward and basic architecture, it has more parameters than all image classification models in this study. It requires the input shape of (224, 224, 3).

Inception-v3: The Inception-v3 architecture was introduced in 2015, and was created to be more accurate than Inception-v1 while using fewer processing resources. Factorized 7x7 convolutions and aggressive spatial pooling are used in Inception-v3, which minimizes the calculation and parameter count. It requires an input shape of (299, 299, 3).

Inception-Resnet-v2: The Inception-Resnet-v2 architecture was released in 2016, and combines the Inception design with residual connections to create deeper and more precise networks. It requires an input shape of (299, 299, 3).

Xception: The Xception architecture was released in 2017. Xception uses depth wise separable convolutions to separate the spatial and channel dimensions in lieu of the standard convolutions to reduce parameters and improve accuracy. It also requires an input shape of (299, 299, 3).

EfficientNet: The EfficientNet architecture was released in 2019. EfficientNet uses the fewer parameters of all image classification models used here and can also provide improved accuracy.

1.4	LSTM and Transformer 
LSTM: Long Short-Term Memory (LSTM) [36] is a Recurrent Neural Network based architecture. LSTMs uses gates to regulate the flow of information into and out of memory cells where they can store information about previous inputs that can be selectively updated. It was first introduced in 1997.

Transformers: Transformers [35] were first introduced in 2017. Transformers capture long-term dependencies without directly maintaining a memory cell because they process entire sequences at once using self-attention mechanisms. In a Transformer, multiple self-attention heads process the input sequence in parallel while dynamically weighing the relative significance of the various segments of the sequence.



2	Evaluation and Results
In fig (6) the generated description with identified disease by all 12 models have been shown. All the encoder decoder models are compared with real anomaly [38] has been shown. In the particular image all of our model seems to provide accurate output. Comparing output of multiple models can also be useful as we can also cross check them.
 
Fig. 6 List of identified disease description generated by all of our model compared to the abnormality [38].
 
Fig. 7 Min validation loss and max accuracy of models in Covid Radiographic Database.
Shown in fig (7) is the min validation loss and max accuracy of models in Covid Radiographic Database. We have calculate validation loss and accuracy for the models which we ran for COVID-19 Radiographic dataset with the epoch set to 15. From fig (10) we can clearly see that, of all trained model the model which provided the least minimum validation loss was Xception-Transformer with batch normalization and max pooling, with a validation loss score of 0.096, and Inception-Resnet-v2-LSTM  based model with batch normalization and max pooling provided highest accuracy for the dataset, with an accuracy of 0.86560. Fig (8) shows the line chart of validation loss and accuracy for all the implemented models from the first epoch to the last.

 
Fig. 8 Line Chart of validation loss for all the implemented models from the first epoch to the last and their epoch value.
 

Fig. 9 Line Chart of accuracy for all the implemented models from the first epoch to the last and their epoch value.


Conclusion
The following work includes a comparative study which we prepared on multiple Encoder-Decoder based disease identification model for chest Xray radiological images, that extracts feature from the images and embeds them using image classification CNNs, the models used in this research are, Xception; Inception-V3; VGG16; Inception-Resnet-V2; and EfficientNetB7 which is then encoded to context vector which then along with the captions provided as input to the decoder which generates captions for those images. We had shown that model with inception-resnet-v2 performed better with inception-resnet-v2-LSTM providing highest accuracy for Covid Radiographic Dataset. While the xception-transformer model provided the least validation loss. Validation loss is a statistic used to assess a machine learning model's generalization capacity. It computes the difference between a model's projected and actual output using data that the model did not see during training. In machine learning tasks, a lower validation loss suggests greater generalization performance. By training the models with higher number of epoch, it can provide more better results. Future work can also benefit from using more sophisticated methods. More sophisticated methods can be used for feature extraction like using object detection models like YOLO [40] for Chest Xray dataset or using Regional Proposal Network (RPN) [41] for focusing on important locations in the Xray images, U-nets [42] can also be used as it can simplify complex structures.

References
1. Yang, S., Wu, X., Ge, S., Zhou, S. K., & Xiao, L. (2022). Knowledge matters: Chest radiology report generation with general and specific knowledge. Medical Image Analysis, 80 , 102510.
2. Selivanov, A., Rogov, O. Y., Chesakov, D., Shelmanov, A., Fedulova, I., & Dylov, D. V. (2023). Medical image captioning via generative pretrained transformers. Scientific Reports, 13 (1), 4171.
3. Lee, H., Cho, H., Park, J., Chae, J., & Kim, J. (2022). Cross encoder-decoder transformer with global-local visual extractor for medical image captioning. Sensors, 22 (4), 1429.
4. Shin, H. C., Roberts, K., Lu, L., Demner-Fushman, D., Yao, J., & Summers, R. M. (2016). Learning to read chest x-rays: Recurrent neural cascade model for automated image annotation. In Proceedings of the IEEE conference on computer vision and pattern recognition(pp. 2497-2506).
5. Wang, X., Peng, Y., Lu, L., Lu, Z., & Summers, R. M. (2018). Tienet: Text-image embedding network for common thorax disease classification and reporting in chest x-rays. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9049-9058).
6. Singh, S., Karimi, S., Ho-Shon, K., & Hamey, L. (2019, December). From chest x-rays to radiology reports: a multimodal machine learning approach. In *2019 Digital Image Computing: Techniques and Applications (DICTA)* (pp. 1-8). IEEE.
7. Allaouzi, I., Ben Ahmed, M., Benamrou, B., & Ouardouz, M. (2018, October). Automatic caption generation for medical images. In *Proceedings of the 3rd International Conference on Smart City Applications* (pp. 1-6).
8. Babar, Z., van Laarhoven, T., & Marchiori, E. (2021). Encoder-decoder models for chest X-ray report generation perform no better than unconditioned baselines. *Plos one*, *16*(11), e0259639.
9. Puscasiu, A., Fanca, A., Gota, D. I., & Valean, H. (2020, May). Automated image captioning. In 2020 IEEE International Conference on Automation, Quality and Testing, Robotics (AQTR) (pp. 1-6). IEEE.
10. Zeng, X., Wen, L., Liu, B., & Qi, X. (2020). Deep learning for ultrasound image caption generation based on object detection. Neurocomputing, 392, 132-141.
11. COVID-19 Radiography Database. (2022, March 19). Kaggle. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
12. De Falco, I., De Pietro, G., & Sannino, G. (2022). Classification of Covid-19 chest X-ray images by means of an interpretable evolutionary rule-based approach. Neural Computing and Applications, 1-11.
13. Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing image description as a ranking task: Data, models and evaluation metrics. Journal of Artificial Intelligence Research, 47, 853-899.
14. Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2097-2106).
15. Elshami, W., Akudjedu, T. N., Abuzaid, M., David, L. R., Tekin, H. O., Cavli, B., & Issa, B. (2021). The radiology workforce's response to the COVID-19 pandemic in the Middle East, North Africa and India. Radiography , 27(2), 360-368.
16. Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1251-1258).
17. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).
18. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
19. Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.
20. Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2017, February). Inception-v4, inception-resnet and the impact of residual connections on learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 31, No. 1).
21. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9 (8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735.
22. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
23. Prechelt, L. (2012). Early stopping—but when?. Neural networks: tricks of the trade: second edition, 53-67.
24. How Does COVID-19 Appear in the Lungs? (2021, October 3). Imaging Technology News. https://www.itnonline.com/content/how-does-covid-19-appear-lungs
25. Vedantam, R., Lawrence Zitnick, C., & Parikh, D. (2015). Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4566-4575).
26. Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.
27. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28.
28. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.



![image](https://github.com/Subhrato20/Encoder-Decoder-based-Chess-Disease-Identification-Model-using-Xray-Radiological-Images/assets/61539946/f92ad3fa-59cc-4111-8051-cdaec3ce694f)
