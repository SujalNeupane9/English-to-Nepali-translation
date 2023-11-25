# English-to-Nepali-translation

If you want to play around with the notebook, click the following button.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Pa38-UVZBteYpns1RbpEvB9nIA5YjQep)


### About the code
This repository contains code for translating English text into Nepali using the M2M100 model from Hugging Face's Transformers library. The model is fine-tuned for sequence-to-sequence (seq2seq) translation tasks.

### Requirements
- Python 3.x
- PyTorch
- Transformers
- Datasets

### Code overview

- English_to_Nepali_translation.ipynb: Contains code for training the seq2seq model and performing English to Nepali translations.

### Example

```bash
english_text = "I am going to the store to buy some groceries."

# Tokenize the input English text using the tokenizer
inputs = tokenizer(english_text, return_tensors='pt')

# Generate the Nepali translation of the input text using the trained model
generated_tokens = model.generate(inputs['input_ids'].to("cuda"),
                                  attention_mask=inputs['attention_mask'].to("cuda"),
                                  num_beams=4,
                                  max_length=512)

# Decode the generated tokens to get the final Nepali translation
nepali_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print("Nepali Translation:", nepali_translation)
```

### Acknowledgement
- The code uses the M2M100 model by Alirezamsh from the Hugging Face Model Hub.
- The code for computing BLEU score utilizes the sacreBLEU metric.
