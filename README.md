# Subjective Answer Evaluation System

This web-based application evaluates scanned subjective answer sheets by comparing student responses with model answers using Optical Character Recognition (OCR) and Natural Language Processing (NLP) techniques. It supports PDF file uploads for both student and model answer sheets.

![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Features

- **Upload PDF Answer Sheets**: Upload scanned PDF files containing student answers.
- **OCR for Text Extraction**: Extracts text from the images in the PDF using Tesseract OCR.
- **Text Comparison**: Compares the extracted text with model answers to evaluate accuracy.
- **Model Accuracy Score**: The application calculates an accuracy score based on the comparison between the student's answer and the model answer.
  
## Getting Started

To run this project locally, follow these instructions:

### Prerequisites

- Python 3.x
- Tesseract OCR
- Required Python libraries (listed in `requirements.txt`)

### Installing Dependencies

1. Clone the repository:

    ```bash
    git clone https://github.com/Praveen-koujalagi/Subjective-Answer-Evaluation-System.git
    cd Subjective-Answer-Evaluation-System
    ```

2. Create a virtual environment:

    ```bash
    python -m venv env
    ```

3. Activate the virtual environment:
    - On Windows:
        ```bash
        .\env\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source env/bin/activate
        ```

4. Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the application, use:

```bash
streamlit run app.py
```

## Team 

- **Praveen Koujalagi** 
- **S Sarvesh Balaji** 
- **Sujit G**
