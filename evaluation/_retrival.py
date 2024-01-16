from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize


def TF_IDF():
    """TODO: Embedding algorithm test
       TODO: Chunking algorithm test
       FIXME: Relevant Chunk test

       # Bag-of-Words (BoW)
       # Term Frequency-Inverse Document Frequency (TF-IDF)
       # Word Embeddings (e.g., Word2Vec, GloVe, FastText)
       # CJIMLOMG TEST
    """
    # Sample documents for retrieval
    documents = [
        "Natural language processing is a subfield of artificial intelligence.",
        "Machine learning algorithms help in building intelligent systems.",
        "Tokenization is an important step in natural language processing.",
        "Recurrent neural networks are used in sequence modeling tasks."
    ]

    # Input prompt
    prompt = "What are the key steps in natural language processing?"

    # Step 1: Tokenize and vectorize documents and prompt
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(documents)
    prompt_vector = vectorizer.transform([prompt])

    # Step 2: Retrieval - Find the most relevant document
    similarities = cosine_similarity(prompt_vector, document_vectors)[0]
    most_similar_index = similarities.argmax()
    most_similar_document = documents[most_similar_index]

    print("Most relevant document:", most_similar_document)

    # Step 3: Relevant Chunk Test - Extract relevant chunks from the most similar document
    sentences = sent_tokenize(most_similar_document)
    relevant_chunks = sentences[:min(2, len(sentences))]  # Extract the first 2 sentences as relevant chunks
    print("Relevant Chunks:", relevant_chunks)

    # Step 4: Embedding Test - Use word embeddings (this is just a placeholder and may require pre-trained embeddings)
    word_embeddings = {
        "natural": [0.1, 0.2, 0.3],
        "language": [0.4, 0.5, 0.6]
    }

    # Calculate the average embedding for the prompt
    prompt_embedding = [word_embeddings[word] for word in prompt.lower().split() if word in word_embeddings]
    average_prompt_embedding = [sum(dim) / len(dim) for dim in zip(*prompt_embedding)] if prompt_embedding else None

    print("Average Prompt Embedding:", average_prompt_embedding)


TF_IDF()