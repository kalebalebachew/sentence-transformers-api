# Embedding Service API

This is a simple FastAPI server that uses the [Sentence Transformers](https://www.sbert.net/) model (`all-mpnet-base-v2`) to generate text embeddings. It’s designed to be used as a microservice for applications like matching systems, search, and recommendation engines.

## Features

- **Generate Embeddings:** POST text to the `/embed` endpoint and get back a vector embedding.
- **Easy to Deploy:** Runs locally or on any cloud platform.
- **Fast & Lightweight:** Built with FastAPI and Uvicorn for high performance.

## Setup

1. **Clone the repository**
2. **Create and activate a virtual environment**
   
```bash
python -m venv venv
``` 
# On Linux/Mac
```
source venv/bin/activate
```
# On Windows
```bash
venv\Scripts\activate
```
3. **Install the dependencies**
```bash
pip install -r requirements.txt
```
## Running the server
```bash
uvicorn main:app --reload
```
## API Usage
### Generate Embedding
**Endpoint: POST /embed**

***Request Body***
```json
{
  "text": "Explain how AI works"
}
```
***Successful response***
```json
{
  "embedding": [0.123, -0.456, ... ]
}
```
## How it works
**Text-to-Vector Conversion**
- The model processes your text and returns an array of numbers (vector embedding) that encodes the semantic meaning of the text.

**Use Cases**

These vector embeddings can be used to :-
- Compare text similarity
- Perform matching between user queries and available content or profiles



