import streamlit as st  # Streamlit library for web apps
import numpy as np  # NumPy for array and image processing
from PIL import Image  # PIL for image loading and saving
import pickle  # For loading CIFAR-10 dataset
import io  # For in-memory file handling

# CIFAR-10 LOADING
@st.cache_data  # Cache data so it loads only once unless changed
def load_cifar10_images():
    with open('data_batch_1', 'rb') as fo:  # Open CIFAR-10 file; reads binary and gives the file object the name fo
        batch = pickle.load(fo, encoding='bytes')  # Load binary file
    data = batch[b'data']  # Get image data
    return data.reshape(-1, 3, 32, 32).astype(np.uint8)  # Reshape to (10000, 3, 32, 32)

# TEXT <-> BITS 
def text_to_bits_with_header(text):
    msg_bytes = text.encode('utf-8')  # Convert text to bytes
    header = [int(bit) for bit in format(len(msg_bytes), '032b')]  # Create 32-bit header with message length
    body = [int(bit) for byte in msg_bytes for bit in format(byte, '08b')]  # Convert message bytes to bits
    return header + body  # Concatenate header and body

def bits_to_text_with_header(bits):
    header = bits[:32]  # First 32 bits are the length header
    length = int(''.join(str(b) for b in header), 2)  # Convert header bits to int
    body = bits[32:32 + length * 8]  # Extract bits for the message
    chars = [int(''.join(str(bit) for bit in body[i:i+8]), 2) for i in range(0, len(body), 8)]  # 8 bits per character
    return bytes(chars).decode('utf-8', errors='ignore')  # Decode bits to string

# LSB STEGANOGRAPHY (Least Significant Bit)
def embed_message_in_image(img_np, bits):
    flat = img_np.flatten()  # Flatten image to 1D
    if len(bits) > len(flat):  # Check if message fits
        raise ValueError("Message too large for image.")
    flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits  # Set LSB of each pixel with message bit
    return flat.reshape(img_np.shape)  # Reshape back to original shape

def extract_bits_from_image(img_np, num_bits):
    flat = img_np.flatten()  # Flatten image
    return (flat[:num_bits] & 1).tolist()  # Extract LSBs as a list

# IMAGE UTILS (image utility functions)
def pil_to_np(pil_img):
    pil_img = pil_img.convert("RGB")  # Ensure image is RGB
    return np.array(pil_img).astype(np.uint8).transpose(2, 0, 1)  # Convert to NumPy (3, H, W)

def np_to_pil(img_np):
    return Image.fromarray(np.transpose(img_np, (1, 2, 0)).astype(np.uint8))  # Convert back to PIL Image

# STREAMLIT APP
st.set_page_config(layout="wide")  # Set Streamlit layout to wide
st.title("Image Steganography")  # App title

tab1, tab2 = st.tabs(["Embed Message", "Extract Message"])  # Two tabs for embed and extract

# TAB 1: EMBED
with tab1:
    st.header("Embed a Secret Message")  # Header for embedding tab

    source = st.radio("Choose image source:", ["Random CIFAR-10 Image", "Upload Your Own"])  # Image source selector

    original_img_np = None  # Placeholder for image array

    if source == "Random CIFAR-10 Image":
        images = load_cifar10_images()  # Load CIFAR-10 images
        idx = st.slider("Select CIFAR-10 Image Index", 0, len(images) - 1, 0)  # Index selector
        original_img_np = images[idx]  # Select image by index
        st.image(np.transpose(original_img_np, (1, 2, 0)), caption=f"CIFAR-10 Image #{idx}", width=200)  # Show image

    else:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])  # Image upload
        if uploaded:
            pil = Image.open(uploaded)  # Open uploaded image
            original_img_np = pil_to_np(pil)  # Convert to NumPy format
            st.image(pil, caption="Uploaded High-Res Image", use_column_width=True)  # Show image

    message = st.text_input("Secret message to embed:", "Enter Secret Message (up to 98,000 characters, including all letters, spaces, and punctuation)")  # Text input for secret message

    if st.button("Embed Message") and original_img_np is not None:  # Embed button
        try:
            bits = text_to_bits_with_header(message)  # Convert message to bits
            stego_np = embed_message_in_image(original_img_np.copy(), bits)  # Embed bits in image
            stego_pil = np_to_pil(stego_np)  # Convert stego image to PIL

            st.subheader("Stego Image")  # Header for stego image
            st.image(stego_pil, caption="Image with Embedded Message", use_column_width=True)  # Show stego image

            buf = io.BytesIO()  # Create in-memory buffer
            stego_pil.save(buf, format="PNG")  # Save image to buffer
            byte_img = buf.getvalue()  # Get bytes from buffer

            st.download_button(  # Create download button
                label="Download Stego Image",
                data=byte_img,
                file_name="stego_image.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f" Error embedding message: {e}")  # Show error if embedding fails

# TAB 2: EXTRACT 
with tab2:
    st.header(" Extract Hidden Message")  # Header for extract tab

    uploaded = st.file_uploader("Upload image with hidden message", type=["png", "jpg", "jpeg"], key="extract")  # Upload stego image
    if uploaded:
        try:
            pil = Image.open(uploaded).convert("RGB")  # Open and convert to RGB
            img_np = pil_to_np(pil)  # Convert to NumPy

            header_bits = extract_bits_from_image(img_np, 32)  # Extract first 32 bits (header)
            length = int("".join(str(b) for b in header_bits), 2)  # Decode header to get message length
            total_bits = 32 + length * 8  # Total bits to extract
            all_bits = extract_bits_from_image(img_np, total_bits)  # Extract all bits
            recovered = bits_to_text_with_header(all_bits)  # Convert bits to text

            st.success(" Extracted Message:")  # Show success message
            st.text_area("Recovered message", recovered, height=150)  # Show recovered message in text area
        except Exception as e:
            st.error(f" Error extracting message: {e}")  # Show error if extraction fails
