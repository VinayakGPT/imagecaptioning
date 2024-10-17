import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import CNNtoRNN
from utils import load_checkpoint
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load your trained model
embed_size = 256
hidden_size = 256
vocab_size = 5000  # Adjust based on your vocab size
num_layers = 1

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
checkpoint = torch.load("my_checkpoint.pth.tar")
model.load_state_dict(checkpoint["state_dict"])
model.eval()  # Set the model to evaluation mode

# Define the transform
transform = Compose([
    Resize((356, 356)),
    transforms.RandomCrop((299, 299)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def generate_caption(image):
    # Process the image and generate caption
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(image, None)  # Pass the image through the model
        # Here you need to convert the output to the caption
        # Example: Get the predicted caption
        # Note: You may need to adjust this depending on how your model outputs captions
        caption = decode_caption(output)  # Implement this function based on your decoding logic
    return caption

def decode_caption(output):
    # Implement decoding logic to convert model output to caption
    # Example: This could involve using argmax and converting indices to words using vocab
    # For simplicity, let's assume output is a tensor of shape [1, vocab_size]
    _, predicted = torch.max(output, dim=2)
    # Convert indices to words (you need to implement this)
    return "Predicted caption here"  # Replace with your decoding logic

# Streamlit app interface
st.title("Image Caption Generator")
st.write("Upload an image to generate a caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        caption = generate_caption(image)
        st.write("Generated Caption:", caption)
