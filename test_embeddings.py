import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List available models
models = client.models.list()

# Filter for embedding models
embedding_models = [model for model in models.data if "embedding" in model.id.lower()]

print("Available embedding models:")
for model in embedding_models:
    print(f"- {model.id}")

# Print all models (in case there are no embedding models)
print("\nAll available models:")
for model in models.data:
    print(f"- {model.id}")