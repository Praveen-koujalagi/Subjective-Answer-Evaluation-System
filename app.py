import pytesseract
import cv2
import tempfile
import os
import fitz
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Set Tesseract command path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load NLP model for sentence embeddings
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to preprocess image for OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary

# Function to extract text using OCR
def extract_text(image_path):
    preprocessed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(preprocessed_image)
    return extracted_text

# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_number in range(len(pdf_document)):
        for img_index, img in enumerate(pdf_document[page_number].get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                temp_image.write(image_bytes)
                images.append(temp_image.name)
    return images

# Function to compare answers using sentence similarity
def compare_answers(student_answer, model_answer):
    student_embedding = nlp_model.encode(student_answer, convert_to_tensor=True)
    model_embedding = nlp_model.encode(model_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(student_embedding, model_embedding).item()
    return similarity

# Function to grade answers based on similarity
def grade_answer(similarity):
    if similarity > 0.85:
        return "Excellent"
    elif similarity > 0.7:
        return "Good"
    elif similarity > 0.5:
        return "Needs Improvement"
    else:
        return "Poor"

# Function to calculate accuracy score based on similarity
def calculate_accuracy(similarity):
    return similarity * 100  # Converting similarity to a percentage

# Streamlit UI
st.title("Subjective Answer Evaluation System")

st.header("Upload Model Answer PDF")
model_pdf = st.file_uploader("Model Answer PDF", type="pdf")

st.header("Upload Student Answer PDF")
student_pdf = st.file_uploader("Student Answer PDF", type="pdf")

if model_pdf and student_pdf:
    with st.spinner("Processing PDFs..."):
        # Process model PDF
        model_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        model_temp.write(model_pdf.read())
        model_temp.close()  # Close the file explicitly after writing
        model_images = extract_images_from_pdf(model_temp.name)
        model_text = " ".join([extract_text(img) for img in model_images])

        # Process student PDF
        student_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        student_temp.write(student_pdf.read())
        student_temp.close()  # Close the file explicitly after writing
        student_images = extract_images_from_pdf(student_temp.name)
        student_text = " ".join([extract_text(img) for img in student_images])

        # Compare and Grade
        similarity_score = compare_answers(student_text, model_text)
        grade = grade_answer(similarity_score)
        accuracy_score = calculate_accuracy(similarity_score)

        # Remove temporary files after processing
        os.remove(model_temp.name)
        os.remove(student_temp.name)

    # Display Results
    st.subheader("Evaluation Results")
    st.write(f"**Similarity Score:** {similarity_score:.2f}")
    st.write(f"**Grade:** {grade}")
    st.write(f"**Model Accuracy Score:** {accuracy_score:.2f}%")
