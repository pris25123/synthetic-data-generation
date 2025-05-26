# synthetic-data-generation


# Real Estate QA Fine-Tuned Language Model

This repository contains the code and resources for fine-tuning the Qwen1.5-0.5B model on a synthetic real estate question-answering (QA) dataset. The goal is to build a specialized AI model capable of answering real estate-related queries accurately and efficiently, with a focus on the Indian real estate context.

## Motivation

Real estate buyers and sellers often encounter complex, region-specific questions about legal processes, documentation, and pricing. This project aims to provide an AI assistant that can respond to these questions, helping users access relevant information quickly and reliably.

## Project Structure

- `RealEstateQA_FineTuning_Colab.ipynb`: Complete Google Colab notebook covering synthetic data generation, model fine-tuning, evaluation, and visualization.  
- `README.md`: This file.

## Usage

1. Open the `RealEstateQA_FineTuning_Colab.ipynb` notebook in Google Colab.
2. login to hugging face to upload the created dataset if you want to.
3. Follow the cells to generate synthetic data, fine-tune the Qwen1.5-0.5B model, and evaluate its performance.  
4. To use the fine-tuned model via Hugging Face Transformers:

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="foreseeitwithme/real-estate-qa-synthetic")
result = qa_pipeline({"question": "What documents do I need to buy a flat?", "context": ""})
print(result)
````

## Results

* **BERTScore F1:** 0.8591
* **Precision:** 0.8549
* **Recall:** 0.8638

The fine-tuned model demonstrates strong performance on synthetic real estate QA tasks.

## Future Work

* Incorporate real-world user question-answer pairs to improve model robustness.
* Expand dataset to cover additional regional terminology and scenarios.
* Experiment with larger models and longer fine-tuning schedules.
* Enhance evaluation with human-in-the-loop feedback.

## References

* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* [BERTScore Paper](https://arxiv.org/abs/1904.09675)
* [Synthetic Data Generation for NLP](https://arxiv.org/abs/2107.07430)

