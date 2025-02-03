# **Run DeepSeek-R1-Distill-Qwen-1.5B in your Local Machine Without Internet Acces**

In this blog post, we’ll walk through the process of setting up your environment, downloading a model from Hugging Face, and running it locally using Python. We’ll use the transformers library and a simple script to demonstrate how easy it is to get started with AI models. Let’s dive in!

## **Step 1: Prepare Your Environment**

Before we start, we need  download the model. Replace `deeseek-ai/deepseek-r1:1.5B` to set up a clean Python environment and install the necessary libraries.

### **1.1. Create a Virtual Environment**

A virtual environment helps keep your project dependencies isolated. Run the following commands in your terminal:


```bash

# Create a virtual environment
$ python -m venv inference_env

# Activate the virtual environment
$ source inference_env/bin/activate  # Linux/Mac
$ inference_env\Scripts\activate     # Windows

```


### **1.2. Install Required Libraries**

Install the `transformers` library and its dependencies (like `torch` for PyTorch):

```bash

$ pip install transformers torch

```


### **2.2. Download the Model**

Use the `huggingface-cli` to download the model. Replace `deepseek-ai/deepseek-r1:1.5b` with the model you want to download:

```bash

$ huggingface-cli download deepseek-ai/deepseek-r1:1.5b --local-dir ./deepseek-r1:1.5b

```

This will download the model to the `./deepseek-r1:1.5b` directory.
## **Step 3: Write the Inference Script**

Now that the model is downloaded, let’s write a Python script to load and run it.

### **3.1 Create `inference.py`**
Create a file named `inference.py` and add the following code:

```Python

# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the downloaded model
model_path = "./deepseek-r1:1.5b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad_token_id to eos_token_id explicitly
tokenizer.pad_token_id = tokenizer.eos_token_id

# Example usage
input_text = "What is AI?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

### **3.2. Explanation of the Code**
- `AutoTokenizer`: Loads the tokenizer for the model, which converts text into tokens (numbers) that the model can understand.

- `AutoModelForCausalLM`: Loads the model for causal language modeling (text generation).

- `pad_token_id`: Ensures the model uses the end-of-sequence token (`eos_token_id`) for padding, which is required for text generation.

- `model.generate()`: Generates text based on the input prompt.
## **Step 4: Run the Script**

With the script ready, it’s time to run it and see the model in action.

### **4.1 Run the Script**

In your terminal, run:


```bash

$ python inference.py

```

### **4.2 Expected Output**

If everything is set up correctly, you’ll see the model’s response to the input prompt `"What is AI?"`. For example:

```bash
AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These systems can perform tasks such as problem-solving, decision-making, and language understanding.
```
## **Step 5: Customize and Experiment**

Now that you have the basic setup working, you can:

- **Change the Input Prompt:** Modify the `input_text` variable to ask different questions or provide new prompts.

- **Experiment with Models:** Replace `deepseek-r1:1.5b` with other models from Hugging Face.

- **Fine-Tune the Model:** Use your own dataset to fine-tune the model for specific tasks.


# **Conclusion**
Running a Hugging Face model locally is straightforward once you’ve set up your environment and downloaded the model. With just a few lines of code, you can load the model, generate text, and start experimenting with AI. Whether you’re building a chatbot, exploring language models, or just curious about AI, this guide provides a solid foundation to get started.

Happy coding! 🚀
