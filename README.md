# Subjective-Answer-Evaluation-System

This system evaluates subjective answers by comparing the student's response to a model answer using Natural Language Processing (NLP) and Optical Character Recognition (OCR) techniques.

## Features

- OCR integration for text extraction from scanned PDF files.
- NLP-based similarity scoring between student and model answers.
- Grading system based on similarity score.

## Requirements

- Python 3.x
- Libraries:
    - `sentence-transformers`
    - `PyMuPDF`
    - `pytesseract`
    - `opencv-python-headless`
    - `streamlit`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Praveen-koujalagi/Subjective-Answer-Evaluation-System.git
    cd Subjective-Answer-Evaluation-System
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Run the following command to start the app:

```bash
streamlit run app.py
