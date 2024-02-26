# Streamlit NLP app - Festival di Sanremo 2024
NLP on Festival di Sanremo 2024 - youtube RAI videos' comments.

This Streamlit application performs Natural Language Processing (NLP) on YouTube comments retrieved via the Google API related to the participants of the Festival di Sanremo 2024. The application provides visualizations and analyses including word clouds, n-grams, sentiment analysis, language distribution, and a summary of comments.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Features

- **Easy filters**: filter the dataset for a specific singer, write stop words, and choose n-grams.
- **Word Cloud:** Visual representation of frequently occurring words in the comments.
- **N-grams Analysis:** Analysis of word combinations (bi-grams, tri-grams, etc.) to identify common phrases.
- **Sentiment Analysis:** Determination of the sentiment (positive, negative) of the comments.
- **Language Distribution:** Distribution of comments based on detected languages.
- **Summary:** Brief overview from the comments.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/pitbull36/streamlit_sanremo24.git
   cd streamlit_sanremo24

   pip install -r requirements.txt

   ```

## Usage

Run the Streamlit App:

```bash

streamlit run sanremo24_app.py

```
## License

This project is licensed under the MIT License - see the LICENSE file for details.