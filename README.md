# IMDB Sentiment Analysis with Hugging Face Transformers

This notebook demonstrates how to build and train a sentiment analysis model for IMDB movie reviews using the Hugging Face Transformers library.

## Project Overview

The goal of this project is to classify movie reviews as either positive or negative. We use a pre-trained `distilbert-base-uncased` model from Hugging Face and fine-tune it on the Stanford IMDB movie review dataset.

## Setup and Installation

To run this notebook, you need to install the following libraries:

```bash
!pip install transformers
!pip install datasets
Dataset
The stanfordnlp/imdb dataset is loaded using the datasets library. It consists of 50,000 movie reviews, split into 25,000 for training and 25,000 for testing.

Each review is labeled as either 0 (negative) or 1 (positive).

Model Training
We fine-tune the distilbert-base-uncased model for sequence classification using the AutoModelForSequenceClassification class and the Trainer API from Hugging Face Transformers. Key training parameters include:

Learning Rate: 2e-5
Batch Size: 16 per device
Number of Epochs: 1
Output Directory: /content/drive/MyDrive/imdb_bert_results (requires Google Drive mounting)
Evaluation
After training, the model's performance is evaluated on the test set. The eval_loss after one epoch was approximately 0.178.

Prediction Function
A predict function is defined to take a raw text review, tokenize it, and return the predicted sentiment (Positive or Negative) along with the confidence scores for each class.

Example Usage:
review = "This movie was absolutely fantastic. The acting was superb!"
label, probs = predict(review)
print(f"Review: {review}")
print(f"Prediction: {label}")
print(f"Confidence: negative={probs[0]:.4f}, positive={probs[1]:.4f}")
Output:

Review: This movie was absolutely fantastic. The acting was superb!
Prediction: POSITIVE 😊
Confidence: negative=0.0043, positive=0.9957
Saving the Model
The trained model and tokenizer are saved to Google Drive at /content/drive/MyDrive/imdb_bert_final for future use.

save_path = "/content/drive/MyDrive/imdb_bert_final"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
Conclusion
This notebook provides a complete pipeline for fine-tuning a BERT-based model for sentiment analysis, from data loading and preprocessing to training, evaluation, and inference.

