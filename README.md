# Optimizing Visual Question Answering with Question-Aware Captioning



A Visual Question Answering (VQA) system that leverages question-driven image captions to improve answer accuracy and explainability.

## Project Overview

This project explores the development of an image caption-based question answering system that leverages pre-existing image captions to accurately respond to user queries. Our approach utilizes natural language processing techniques, particularly focusing on models like BERT for keyword extraction and semantic understanding.

Key features:
- Question-driven caption generation using KeyBERT and CogVLM
- Integration with large language models (GPT-3.5) for answer generation
- Improved accuracy in zero-shot VQA scenarios
- Enhanced explainability through intermediate caption outputs

## Results

Our approach achieves:
- 49.50% overall accuracy on GQA dataset
- 66.83% accuracy on Yes/No questions
- 59.07% accuracy on Logical Reasoning tasks

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Question-Aware-VQA.git
cd Question-Aware-VQA
