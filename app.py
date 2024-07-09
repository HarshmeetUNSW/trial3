import streamlit as st
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import plotly.graph_objs as go

# Load an image
logo = Image.open("TikTok-logo.png")

# Set up the layout
col1, col2 = st.columns([1, 3])
with col1:
    st.image(logo, width=100)  # Adjust width as needed

with col2:
    st.title('Hate Speech Detector')

# Load the model and tokenizer once
@st.cache_data
def load_model():
    print('Once')
    model = BertForSequenceClassification.from_pretrained('final_fine_tuned_bert/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.eval()  # Put the model in evaluation mode
    return model, tokenizer

model, tokenizer = load_model()

# Initialize LIME Text Explainer
explainer = LimeTextExplainer(class_names=["Hateful", "Normal", "Offensive"])

def predict_and_explain(text):
    # Check if the input text is empty or too short
    if not text.strip():  # This checks for empty or whitespace-only strings
        st.warning("Please enter a valid text for prediction.")
        return None, None, None

    try:
        # Prepare the text input for BERT
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180)

        # Disable gradient calculation
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predictions
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        max_prob, predicted_label = torch.max(probabilities, dim=1)

        # Generate LIME explanation
        exp = explainer.explain_instance(text, 
                                         lambda x: torch.softmax(model(**tokenizer(x, return_tensors='pt', padding=True, truncation=True)).logits, dim=1).detach().numpy(),
                                         num_features=6,
                                         num_samples=100,  # Adjust the number of samples for faster execution if needed
                                         top_labels=1)

        # Convert LIME explanation to Plotly figure (horizontal bar chart)
        exp_list = exp.as_list(label=predicted_label.item())
        features = [x[0] for x in exp_list]
        importances = [x[1] for x in exp_list]
        colors = ['green' if x > 0 else 'red' for x in importances]
        
        fig = go.Figure([go.Bar(x=importances, y=features, orientation='h', marker_color=colors)])
        fig.update_layout(title='Word Importance Towards The Final Output',
                          xaxis_title='Importance',
                          yaxis_title='Words',
                          yaxis=dict(autorange="reversed"))  # Ensure that the most important feature appears on top

        return predicted_label, max_prob.numpy()[0], fig
    except Exception as e:
        st.error(f"Prediction or explanation failed: {e}")
        return None, None, None


def get_emoji(label):
    if label == 0:
        return "😡", "Hateful"  # Update label descriptions as per your labels
    elif label == 1:
        return "😊", "Normal"
    else:
        return "😠", "Offensive"

def on_predict():
    text = st.session_state.text
    if text:
        label, confidence, fig = predict_and_explain(text)
        if label is not None:
            emoji, label_description = get_emoji(label)
            confidence = round(confidence * 100, 2)
            st.session_state.results = f'{emoji}  {label_description} with Confidence: **{confidence}%**'
            st.session_state.fig = fig

# Text input
st.text_area("Enter Text for Prediction", key="text", on_change=on_predict)

# Predict button
st.button("Predict", on_click=on_predict)

# Display results
if "results" in st.session_state:
    st.markdown(st.session_state.results, unsafe_allow_html=True)
    st.plotly_chart(st.session_state.fig)
