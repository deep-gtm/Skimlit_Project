# Skimlit_Project

This proejct helps in text classification of abstract of medical paper. A text written in categorized form is much more easy to read as compared to when written as just as a chunk of text.
- You can see on the right hand side text is classified in Background, Method, Results and Conclusion

![what we are doing](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/d46f61cc-c88b-46fa-aaad-43d065797cad)

# Objective :

- Main Aim of this project is to build a NLP model so that it can label Abstract of a medical research paper.
- The paper that we are replicating the source of dataset that we will be using is available [here](https://arxiv.org/abs/1710.06071)
- Reading paper we see the model architecture that they use to achieve best score is available [here](https://arxiv.org/abs/1612.05251)
- Dataset has been made publicly available [here](https://github.com/Franck-Dernoncourt/pubmed-rct)
- I am working out experiment with 20k dataset with numbers replace by @ sign

# Distribution of labels in Abstract in Pubmed 20K RCT Dataset

- These will be labels that we will be classigying our text in Abstract of medical research paper.
![distribution of labels in abstract](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/6b0ea23e-3b0f-483d-bedf-693bdd107e1c)

# Preprocessing
- Data that was given in dataset need to preprocessed so that we can use it for training of our models.
- In this i took every line from abstract and assign it it's line number in abstract, total lines in abstract to whih it belong to and text and label for that line
- For converting target labels to numeric values i used sklearn OneHotEncoder, LabelEncoder(for integer encoding).
  
![data preprocessing](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/49a7ceba-39fb-4159-8fa4-95f6b998b0ea)

# Models

- In this project i tried different machine learning model that we can use for our classfication and at last implemented model that is used in [paper](https://arxiv.org/abs/1612.05251)
0. Model 0 : Naive Bayes with TF-IDF Encoder (baseline)
1. Model 1 : Conv1D with Token Embeddings.
2. Model 2 : Tensorflow Hub Feature Extractor (Universal Sentence Encoder).
3. Model 3 : Conv1D with Character Embedding.
4. Model 4 : Pretrained Token Bmbedding + Character Embedding
5. Model 5 : Pretrained Token Embedding + Character Embedding + Positional Embedding (Model described in paper)

# Results of different model
**Note** : Because of unavailability of resources for training of model i was not able to train on whole dataset insted i trained and validated my every model on 10% of train and val dataset and also i trainded model only for 5 epochs. However we can compare results and see which worked better :

## Model 0 (Naive Bayes TF-IDF) 72% Accuracy
In Naive Bayes TF-IDF, the TF-IDF values of words are used as features for the Naive Bayes classifier. Here's how the process works:

- Text Preprocessing: The text data is preprocessed, which typically involves tokenization, removing stop words, stemming/lemmatization, and other text cleaning steps.

- TF-IDF Calculation: For each document, TF-IDF values are calculated for each word in the vocabulary. TF-IDF values are computed using the formula: TF-IDF = (Term Frequency) * (Inverse Document Frequency)

- Naive Bayes Classification: The TF-IDF values for each document are used as input features for the Naive Bayes classifier. The classifier calculates the probabilities that a document belongs to each class based on the TF-IDF features.

- Prediction: The class with the highest probability is assigned as the predicted class for the document.

Naive Bayes TF-IDF can be a powerful approach for text classification tasks, especially when you have limited labeled data. It combines the probabilistic nature of Naive Bayes with the ability of TF-IDF to capture the uniqueness of words in documents, allowing the model to make informed predictions based on word frequencies and document relationships.

- Classification Metrics:
  
![Screenshot 2023-08-13 at 3 48 55 PM](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/5f61a34d-ee8a-4868-8b7f-738f69680065)

- Confusion Matrix

![model_0_confusion_matrix](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/056367f0-0d3e-4af4-85bb-215d31ffb2c7)


## Model 1 (Custom Token Embedding with Conv1D layer) 80% Accuracy
- Achiever **80% accuracy** (better than baseline model)
- Convolutional Neural Networks (ConvNets or CNNs) are commonly used to process sequential data like text.
- Input Encoding: The input to the Conv1D layer is typically a sequence of word embeddings or characters. **(word2vec)**
- Convolution Operation: The Conv1D layer performs a convolution operation across the input sequence. In NLP, this operation involves sliding a small filter ver the input sequence to capture local patterns. The filter is a small window of trainable weights.
- Pooling: After the convolution operation, a pooling layer (often MaxPooling) is applied to reduce the dimensionality of the feature maps. Pooling involves selecting the maximum value within a window, further abstracting and condensing the extracted features.
- Flattening: The pooled feature maps are typically flattened to convert them into a one-dimensional vector. This vector contains the extracted features from the original input sequence.
- Fully Connected Layers: The flattened feature vector is then passed through one or more fully connected layers for classification.
- Output: The final layer produces class probabilities or scores for the different classes in your classification task. Softmax activation is often used to convert scores into probabilities.
- Exposure : TextVectorization, Embedding, Conv1D

- Classification Metrics
  
![Screenshot 2023-08-13 at 4 07 27 PM](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/47c00e95-a853-4018-bc5e-19e1c9943137)

- Confusion Matrix

![model 1 confusion matrix](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/458bc956-e5f3-4df5-b60f-edda1e3995dd)

## Model 2 (Transfer Learning) 75% Accuracy

- Used Universal Sentence Encoder for token embedding
- Provided by [tensorflow hub](https://tfhub.dev/google/universal-sentence-encoder/4)
- Achieved **75% accuracy** (maybe if train for more epochs will get more accuracy)
- Exposure : Transfer learning, tensorflow hub, hub.KerasLayer
- Classificaiton Metric
  
![Screenshot 2023-08-13 at 4 21 01 PM](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/d634684f-706e-4824-8254-86447e94e803)

- Confusion Matrix

![model 2 confusion matrix](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/cddb7e1e-65e4-4f22-9d7c-11e0b8003ff9)


## Model 3 (character level embedding with conv1D) 47% Accuracy
- used character wise embedding
- achieved **47% accuracy** (lowest accuracy achieved with this)
- Exposure : Bidirection, LSTM
- Classification metric

  ![Screenshot 2023-08-13 at 4 26 34 PM](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/63e2c45d-29fc-457d-9860-aa98216d9b5b)

- Confusion Matrix

  ![model 3 confusion matrix](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/d2dc4c9d-6ef8-44cd-8a28-f0478ec967bd)


## Model 4 (Character + Token) 74% Accuracy

- In this used character level as well as token level embedding for our input sentences
- for character level used TextVectorization and Embedding layer
- for token level used feature extractor (transfer learning, tensorflow hub, universal sentence encoder)
- Also i learned how to work with multimodal and have to pass multiple inputs to our model
- Exposure : Multimodel, Concetanate (used for concatenation of character and token embedding)
- Model 4 Visualization :

  ![model 4 visual representation](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/f61333d7-dbe8-4a84-b4d1-b72bd5828891)


- Achieved **74% accuracy** (may increase if train for more epoch)
- Classification Metrics

  ![Screenshot 2023-08-13 at 4 39 26 PM](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/3f107f62-991d-49bf-9e32-8007285adb46)

- Confusion Matrix

  ![model 4 confusion matrix](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/7d8f8e8e-fe4d-428b-a51e-b0adf08f0070)


# Model 5 (char + token + position) 84% Accuracy

- This is the model that was described in the paper
- In this we also take care of position of embedding
- As it is important because sentences in an abstract which appear earlier are most likely to belong ot Background or Objective and sentenced at the end are most likely to belong to results and conclusion.
- So this is why for creating position embedding we took care of line number and total lines in abstract for our sentences.
- This is an example of **feature engineering** : it is converting non obvious information from data to useful one that can help in training of our model and help in achieving better results.
- For converting line number and total_line into position embedding we used tf.one_hot(), that is one hot encoder
- Achieved **84% accuracy** (Maximum among others)
- Visual Representation Of Model :

![model 5 visual representation](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/71fba438-a2a1-4b6b-9d19-8541e4c55a0a)

- Classification Metrics

  ![Screenshot 2023-08-13 at 4 50 25 PM](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/6f64b020-5005-40e1-a3be-df9ba56ba70e)

- Confusion Matrix

  ![model 5 confusion matrix](https://github.com/deep-gtm/Skimlit_Project/assets/70434931/c370f20d-40de-438d-9539-24bf4af9994d)


# Conclusion
- We can see most of the timse model get confused between objectives and background
- Our Model 5 workded better than other models
- We also learned how positional encoding is much useful feature in text classfication as in model 5
- We also learned how to work with sequential data, got to know about embeddings, convolution layer in sequetial data and much more.
