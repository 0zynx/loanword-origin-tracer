# Loanword Origin Tracer

A language-origin analysis tool that combines machine learning with etymological data to explore word origins, language relationships, and borrowing patterns.

## Overview

Loanword Origin Tracer analyses words using trained language-classification models and a processed linguistic dataset. It provides predicted language labels alongside recorded etymological information to help users investigate whether words are associated with borrowing or inheritance.

The project separates machine-learning predictions from documented linguistic evidence, since a model prediction alone cannot establish a word's historical origin.

## Features

- **Language Prediction:** Predicts a likely language using trained machine-learning models.
- **Multiple Input Views:** Supports native-script and transliterated input modes.
- **Prediction Scores:** Displays the top five predicted language scores.
- **Character Influence:** Examines how removing individual characters changes the model's output.
- **Etymology Lookup:** Searches the processed dataset for matching words and recorded origin information.
- **Borrowing Relationships:** Displays recorded source and recipient language information where available.
- **Historical Information:** Presents available etymological notes and historical stages from the dataset.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- PyTorch
- Linguistic and etymological data

## Project Structure

```text
loanword-tracing/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── preds/
│   └── tables/
├── src/
│   ├── app.py
│   ├── analyze.py
│   ├── data.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── models.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   └── train_neural.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python installed
- Git
- pip

### Installation

Clone the repository:

```bash
git clone https://github.com/0zynx/loanword-origin-tracer.git
cd loanword-origin-tracer
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal to use the application.

## Machine Learning

The project includes baseline and neural-network model artifacts for native-script and transliterated input. The application uses the selected trained model to generate language predictions and scores.

Model predictions are exploratory outputs. They should not be treated as definitive evidence of etymology or borrowing.

## Data and Limitations

The application relies on the available processed dataset and trained models. Historical details may be incomplete, and a recorded source language does not necessarily establish every intermediate stage in a word's history.

Dataset coverage, model performance, and source-data licensing should be considered when interpreting results or redistributing project files.

## Future Improvements

- Interactive visualisation of documented word histories
- Clearer exploration of borrowing and inheritance relationships
- Improved prediction evaluation and explanations
- Interface and usability improvements

## Author

**0zynx**

GitHub: https://github.com/0zynx
