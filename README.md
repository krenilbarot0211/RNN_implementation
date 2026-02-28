RNN Based Sentence Classification using TensorFlow & Keras

📌 Project Overview

This project demonstrates a simple yet powerful Recurrent Neural Network (RNN) model built using TensorFlow and Keras for binary sentence classification.

The model is trained on a small custom dataset (~30 sentences) and learns to classify text into two categories using deep learning techniques such as tokenization, embeddings, and sequence modeling.

This project helped me understand:

* How RNN works internally
* Text preprocessing pipeline
* Embedding layers
* Binary classification using Deep Learning



Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Scikit-learn (if used for train-test split)



🏗️ Model Architecture

The architecture of the model:

1. Text Tokenization

   * Convert sentences into sequences of integers
   * Vocabulary creation
   * Padding to ensure equal input length

2. Embedding Layer

   * Converts integer sequences into dense vector representations
   * Learns word relationships during training

3. Simple RNN Layer

   * Captures sequential dependencies in text
   * Learns contextual meaning from previous words

4. Dense Output Layer

   * 1 neuron
   * **Sigmoid activation**
   * Used for binary classification



🧮 Model Summary (Conceptually)

Input → Tokenization → Padding → Embedding → RNN → Dense (Sigmoid)



⚙️ Training Configuration

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Evaluation Metric: Accuracy
* Activation Function (Output): Sigmoid



📊 Why Sigmoid?

Since this is a binary classification problem, sigmoid outputs a probability between 0 and 1:

* Output close to 0 → Class 0
* Output close to 1 → Class 1



📚 Key Learning Outcomes

* Understanding how RNN processes sequential data
* Difference between Bag of Words and sequence-based learning
* Importance of Embeddings in NLP
* Role of activation functions in classification
* How loss functions guide model learning



📂 Project Structure


rnn-text-classifier/
│
├── model_training.ipynb
├── rnn_model.py
├── README.md
└── requirements.txt





📈 Future Improvements

* Increase dataset size
* Use LSTM or GRU instead of Simple RNN
* Add Dropout for regularization
* Deploy using Streamlit
* Save and load trained model
* Add confusion matrix & precision/recall metrics



## ✨ Author

Krenil Barot
AI/ML Enthusiast 🚀


