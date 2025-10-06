# Week 1 Notes

# Jupyter
- Allows you to run pieces of code without running an entire program, playground for learning how to work with LLMs
- Jupyter notebooks (.ipynb files) let you mix code, text, and output in a single place
- Run code cells one at a time instead of all at once
- Great for ML/AI

# Hugging Face
- open source hub for LLM, few lines of python to use LLM
- use `transformers` lib to download LLM with few lines of code
- from `transformers` lib import hugging face `pipeline` API to utilize pre-trained LLM
- syntax to load model: `generator = pipeline("pipeline", model="model")`
    - pipelines: text-generation, sentiment-analysis, summarization, translation_xx_to_xx (en to fr)
    - models
        - decoder only (text-generation): gpt2
        - encoder/decoder seq2seq: google/flan-t5-base
    - single library, different NLP, from pre-trained models
- syntax to run inference: `output = generator("Prompt", max_length=##, num_return_sequences=#)`
    - giving model a prompt
    - predict the next words, until `max length` in tokens
    - how many results you want with `num_return_sequences`
    - show models take input text
    - show how outputs depend on model size, randomness, and prompt
    - can also include `temperature` and `top_k` and `top_p`