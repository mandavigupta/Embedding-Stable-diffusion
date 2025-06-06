import torch
from gensim.downloader import load
from diffusers import StableDiffusionXLPipeline
import torch.nn.functional as F

# Load GloVe embeddings

model = load("glove-wiki-gigaword-300")  # 300D GloVe vectors

# User input
print("\nEnter 3 words for A - B + C analogy:")
word_a = input("Enter word A: ").strip().lower()
word_b = input("Enter word B: ").strip().lower()
word_c = input("Enter word C: ").strip().lower()

for w in [word_a, word_b, word_c]:
    if w not in model:
        raise ValueError(f"Word '{w}' not in GloVe vocabulary!")


print(f"\nComputing: {word_a} - {word_b} + {word_c}")
results = model.most_similar(positive=[word_a, word_c], negative=[word_b], topn=5)
print("\nTop predictions:")
for i, (word, score) in enumerate(results):
    print(f"{i+1}. {word} ({score:.4f})")

# Use top-1 for image prompt
predicted_word = results[0][0]
prompt = f"A {predicted_word} is sitting on the chair"

print("\nLoading Stable Diffusion XL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Generate image
print(f"\nGenerating image for: '{prompt}'")
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("output_image.png")
image.show()
print("Image saved as output_image.png")
