AI-Powered Perfume Recommendation System
 Overview

This project is an intelligent perfume recommendation engine built with Deep Learning. Instead of relying on simple keyword matching or user ratings, the system understands the "chemical DNA" and "semantic profile" of over 26,000 perfumes. It utilizes an Artificial Neural Network (Autoencoder) to compress fragrance notes, main accords, and gender target audiences into a 128-dimensional latent space.
 Features

    Deep Learning Based: Uses a Keras Autoencoder to map categorical textual data into a dense mathematical vector space.

    Semantic Matching: Recommends perfumes based on the overall "vibe" and chemical makeup (Top, Middle, Base notes + Main Accords), not just identical names.

    Cross-Brand Discovery: Includes a custom filtering algorithm to prevent overspecialization (flanker trap), forcing the AI to suggest identical scent profiles from entirely different brands.

 How It Works
Getty Images

    Data Preprocessing: Raw data is cleaned, and text features (Notes, Accords, Gender) are combined and vectorized using CountVectorizer.

    Autoencoder Architecture: The high-dimensional sparse matrix is fed into a Neural Network (Input -> Dense(512) -> Dense(128) [Bottleneck] -> Dense(512) -> Output).

    Embeddings: The 128-dimensional bottleneck layer is extracted as the unique "DNA" (perfume_embeddings.npy) of each fragrance.

    Recommendation: Cosine Similarity is calculated between these embeddings to find the nearest neighbors in the latent space.

 Exported Models & Files

    final_encoder.keras: The trained bottleneck layer (brain) of the neural network.

    perfume_embeddings.npy: The 128-D coordinates for all perfumes for fast distance calculation.

    final_vectorizer.pkl: The dictionary that converts human-readable notes into machine-readable arrays.

    final_cleaned_perfumes.csv: The cleaned dataset for the front-end display.

 Future Work

    Integration with a Flask/Django backend to serve predictions via a RESTful API.

    Building a web-based UI for users to select their favorite perfumes and apply dynamic filters (e.g., target gender, specific brands).
