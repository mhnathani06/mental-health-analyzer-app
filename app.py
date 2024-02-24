import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image

# Download NLTK resources
nltk.download('vader_lexicon')

# Set page title and favicon
st.set_page_config(page_title="Mental Health Analyzer", page_icon=":brain:")

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to preprocess user input
def preprocess_text(text):
    # Apply preprocessing steps as needed
    processed_text = text.lower()  # Example: Convert text to lowercase
    return processed_text

# Function to predict anxiety/depression using VADER
def predict_anxiety_depression(text):
    # Analyze sentiment
    scores = sid.polarity_scores(text)
    # Determine if there are indications of anxiety/depression
    if scores['compound'] <= -0.05:
        image = Image.open("sad.jpg")
        return 'Positive for Anxiety/Depression😭', image
    else:
        image = Image.open("happy.jpg")
        return 'No Indications of Anxiety/Depression😍', image

# Streamlit app
def main():
    # Image
    image = Image.open("vibes.png")
    st.image(image, use_column_width=True)

    # Page title and description
    st.title("Mental Health Analyzer")
    st.write("Welcome to the Mental Health Analyzer. Enter a sentence to analyze for indications of anxiety/depression.")

    # Text input box
    user_input = st.text_area("Enter your sentence:")

    # Analyze button
    if st.button("Analyze"):
        result_label, result_image = predict_anxiety_depression(user_input)
        st.write(f"**Result:** {result_label}")
        st.image(result_image, caption=result_label, use_column_width=True)

    # About section
    st.sidebar.title("About")
    st.sidebar.info(
        "This app analyzes text input to detect indications of anxiety/depression using a trained model and Vader tool.")

if __name__ == "__main__":
    main()
