# Fine-Tuned T5 Model for Generating NoSQL Queries from Natural Language 🚀

This repository contains a T5 model fine-tuned to translate natural language questions into NoSQL (MongoDB) queries, demonstrated with a Pokémon dataset. The model harnesses the power of T5 (Text-to-Text Transfer Transformer) to bridge the gap between human language and database querying languages.

![T5 Model](https://huggingface.co/front/assets/huggingface_logo.svg)

## 🎯 Purpose

The fine-tuned model aims to simplify data querying by allowing users to interact with databases through natural language. This approach is particularly beneficial for non-technical users or for rapid prototyping.

## 📊 Dataset

The Pokémon dataset used for training encompasses a wide range of data points, including Pokémon types, stats, and other attributes, stored in a MongoDB database. The dataset was transformed into a series of natural language questions paired with their corresponding MongoDB queries.

## 🛠 Training

- **Model Base**: `t5-small`
- **Epochs**: 50
- **Batch Size**: 4 (Training), 4 (Evaluation)
- **Warmup Steps**: 500
- **Weight Decay**: 0.01

Training was conducted in Google Colab, leveraging its GPU resources for efficient model optimization.

## 📝 How to Use

1. **Setup Environment**

```bash
pip install transformers torch
```

2. **Load the Model**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('./T5_fine_tuned')
tokenizer = T5Tokenizer.from_pretrained('./t5_fine_tuned')
```

3. **Query Generation**

```python
def generate_query(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True)
    output = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

question = "Find all Fire-type Pokémon"
query = generate_query(question)
print(query)
```

## 📔 Colab Notebook

For a comprehensive guide on the fine-tuning process, model training, and implementation details, refer to our [Google Colab notebook](#).

## 📄 License

This project is licensed under [MIT License](LICENSE).

---

Feel free to explore, modify, and distribute this project. For any queries or suggestions, please open an issue in this repository.
