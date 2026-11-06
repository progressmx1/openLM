from llama_cpp import Llama

# Load your model
llm = Llama(
    model_path="./models/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K_L.gguf",

    n_ctx=8192,        # Context window
    n_gpu_layers=-1,   # Offload all possible layers to GPU
    verbose=False
)

print("Model loaded successfully!")

# Generate a response
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Explain how a neural network works."
        }
    ],

    temperature=0.7,
    max_tokens=1024
)

# Display response
answer = response["choices"][0]["message"]["content"]

print("\nAI Response:")
print(answer)