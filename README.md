# 🧠 GloVe Analogy → Image Generator 🎨  
Turn a word analogy like `king - man + woman` into an AI-generated image using word embeddings and Stable Diffusion.

---

## 🧾 Overview

This project combines **word vector arithmetic** and **text-to-image generation** to visualize semantic analogies.

> Example:  
> `king - man + woman ≈ queen` → _"A queen is sitting on the chair"_ → 🎨 *Image generated!*


## 🛠️ 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/glove-analogy-image-generator.git
cd glove-analogy-image-generator
```

## 📦 2. Install Dependencies



Requires: Python 3.8+, CUDA-enabled GPU

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gensim diffusers transformers accelerate
```
▶️ 3. Run the Script
```bash
python main.py
```
🎯 4. Provide Input Words
You'll be asked to input three words in this analogy format:
```bash
Enter word A: king
Enter word B: man
Enter word C: woman
```
🔍 5. How It Works

>Uses GloVe embeddings (300D vectors) to solve the analogy:

   vec(A)−vec(B)+vec(C)=vec(D)
   
>Passes the predicted word D into a natural-language prompt template

>Feeds the prompt into Stable Diffusion XL to generate a matching image

🤖 Models Used
1. GloVe (Global Vectors for Word Representation)
   
    ->Source: glove-wiki-gigaword-300 via gensim
  
    ->300-dimensional embeddings trained on Wikipedia + Gigaword
  
    ->Captures semantic relationships between words
  
2. Stable Diffusion XL (SDXL)

    ->Model: stabilityai/stable-diffusion-xl-base-1.0
  
    ->High-resolution text-to-image generation
  
    ->Uses Hugging Face diffusers library

   
💡 Prompt Example
The prompt is dynamically generated like so:
```bash
prompt = f"A {predicted_word} is eating the food"
```
🖼️ Sample Output
✅ Input:
```bash
A = king
B = man
C = woman
```
✅ Predicted word:
```bash
queen
```
✅ Prompt used:
```bash
A queen is eating the food
```
🖼️ Generated Image:
![Output Image](https://raw.githubusercontent.com/mandavigupta/Embedding-Stable-diffusion/main/output_image.png)



