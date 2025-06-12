import streamlit as st  # Streamlit for web app UI
import numpy as np  # NumPy for numerical array manipulation
from PIL import Image  # PIL (Pillow) for image loading and saving
import pickle  # For loading CIFAR-10 dataset batches
import io  # For in-memory byte stream handling
import zlib  # For compressing and decompressing message data

st.set_page_config(layout="wide")  # Set Streamlit app layout to wide mode

import hashlib  # For SHA-256 hashing (integrity check)
from cryptography.fernet import Fernet  # For symmetric encryption/decryption (Fernet cipher)

# Encryption Key UI and management
st.sidebar.title("Encryption Key")  # Sidebar title for encryption key display

if "fernet_key" not in st.session_state:  # Check if key is not already stored in session
    st.session_state.fernet_key = Fernet.generate_key()  # Generate a new random Fernet key and store it

cipher = Fernet(st.session_state.fernet_key)  # Create cipher object with current key

st.sidebar.code(st.session_state.fernet_key.decode(), language="text")  # Display the encryption key as text in sidebar

# CIFAR-10 dataset loading with caching
@st.cache_data  # Cache loaded data to avoid reloading on each rerun
def load_cifar10_images():
    with open('data_batch_1', 'rb') as fo:  # Open CIFAR-10 batch file in binary mode
        batch = pickle.load(fo, encoding='bytes')  # Load batch as dictionary with byte keys
    data = batch[b'data']  # Extract raw image data (flat arrays)
    return data.reshape(-1, 3, 32, 32).astype(np.uint8)  # Reshape to (N, 3, 32, 32) and convert to uint8

# Encode text message securely to bits with compression, hashing, encryption, and header
def text_to_bits_with_header_secure(text):
    compressed = zlib.compress(text.encode('utf-8'))  # Compress UTF-8 encoded text using zlib
    hashed = hashlib.sha256(compressed).digest()  # Compute SHA-256 hash of compressed data (32 bytes)
    combined = compressed + hashed  # Append hash as checksum to compressed data
    encrypted = cipher.encrypt(combined)  # Encrypt combined data with Fernet cipher
    header = [int(bit) for bit in format(len(encrypted), '032b')]  # 32-bit header encoding encrypted data length in bits
    body = [int(bit) for byte in encrypted for bit in format(byte, '08b')]  # Convert encrypted bytes to bits
    return header + body  # Return combined header and body bits as a list

# Decode bits to text using a custom cipher (Fernet) and integrity check
def bits_to_text_with_custom_key(bits, cipher):
    header = bits[:32]  # Extract first 32 bits as header for encrypted data length
    length = int(''.join(str(b) for b in header), 2)  # Convert header bits to integer length
    body = bits[32:32 + length * 8]  # Extract encrypted data bits using length from header
    encrypted_bytes = bytes([int(''.join(str(bit) for bit in body[i:i+8]), 2) for i in range(0, len(body), 8)])  # Convert bits back to bytes
    try:
        combined = cipher.decrypt(encrypted_bytes)  # Decrypt encrypted bytes with Fernet cipher
        compressed = combined[:-32]  # Extract compressed message part (all but last 32 bytes)
        checksum = combined[-32:]  # Extract checksum (last 32 bytes)
        if hashlib.sha256(compressed).digest() != checksum:  # Verify checksum integrity
            return "Integrity check failed."  # Return error if checksum mismatch
        return zlib.decompress(compressed).decode('utf-8')  # Decompress and decode message text
    except:
        return "Decryption failed."  # Return error if decryption fails

# Embed bits into image using Least Significant Bit (LSB) steganography
def embed_message_in_image(img_np, bits):
    flat = img_np.flatten()  # Flatten 3D image array to 1D
    if len(bits) > len(flat):  # Check if message fits into image pixels
        raise ValueError("Message too large for image.")  # Raise error if too large
    flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits  # Set LSB of pixels to message bits
    return flat.reshape(img_np.shape)  # Reshape flat array back to original image shape

# Extract specified number of bits from image LSBs
def extract_bits_from_image(img_np, num_bits):
    flat = img_np.flatten()  # Flatten image to 1D
    return (flat[:num_bits] & 1).tolist()  # Extract LSBs as list of bits

# Convert PIL Image to NumPy array in (channels, height, width) format
def pil_to_np(pil_img):
    pil_img = pil_img.convert("RGB")  # Ensure image is in RGB mode
    return np.array(pil_img).astype(np.uint8).transpose(2, 0, 1)  # Convert to NumPy and reorder axes to (C, H, W)

# Convert NumPy array (channels, height, width) back to PIL Image
def np_to_pil(img_np):
    return Image.fromarray(np.transpose(img_np, (1, 2, 0)).astype(np.uint8))  # Transpose to (H, W, C) and convert to PIL

# Streamlit app UI setup
st.title("🔐 Secure Image Steganography")  # Main title of the app

tab1, tab2 = st.tabs(["🖼️ Embed Message", "🔍 Extract Message"])  # Create two tabs for embedding and extraction

# TAB 1: Embed Message UI and logic
with tab1:
    st.header("🔐 Embed a Secret Message")  # Header for embedding tab

    source = st.radio("Choose image source:", ["Random CIFAR-10 Image", "Upload Your Own"])  # Let user choose image source

    original_img_np = None  # Initialize variable for original image array

    if source == "Random CIFAR-10 Image":
        images = load_cifar10_images()  # Load CIFAR-10 images from batch
        idx = st.slider("Select CIFAR-10 Image Index", 0, len(images) - 1, 0)  # Slider to pick image index
        original_img_np = images[idx]  # Select image by index
        st.image(np.transpose(original_img_np, (1, 2, 0)), caption=f"CIFAR-10 Image #{idx}", width=200)  # Display selected image

    else:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])  # Allow user to upload image in png, jpg and jpeg
        if uploaded:
            pil = Image.open(uploaded)  # Open uploaded image as PIL
            original_img_np = pil_to_np(pil)  # Convert PIL image to NumPy format (C, H, W)
            st.image(pil, caption="Uploaded High-Res Image", use_column_width=True)  # Display uploaded image

    message = st.text_input("💬 Secret message to embed:", "Enter Secret Message")  # Input field for secret message

    if st.button("🔐 Embed Message") and original_img_np is not None:  # When embed button clicked and image is available
        try:
            bits = text_to_bits_with_header_secure(message)  # Convert message text to secure bits
            stego_np = embed_message_in_image(original_img_np.copy(), bits)  # Embed bits into image copy
            stego_pil = np_to_pil(stego_np)  # Convert stego NumPy array back to PIL Image
            st.subheader("🔍 Stego Image")  # Subtitle for showing stego image
            st.image(stego_pil, caption="Image with Embedded Message", use_column_width=True)  # Display stego image

            buf = io.BytesIO()  # Create in-memory bytes buffer
            stego_pil.save(buf, format="PNG")  # Save stego image to buffer as PNG
            byte_img = buf.getvalue()  # Get bytes data from buffer
            st.download_button("⬇️ Download Stego Image", data=byte_img, file_name="stego_image.png", mime="image/png")  # Provide download button
        except Exception as e:
            st.error(f"❌ Error embedding message: {e}")  # Show error message if embedding fails

# TAB 2: Extract Message UI and logic
with tab2:
    st.header("🔍 Extract Hidden Message")  # Header for extraction tab

    user_key_input = st.text_input("🔑 Enter the encryption key used during embedding:")  # Input field for encryption key

    uploaded = st.file_uploader("Upload image with hidden message", type=["png"], key="extract")  # Upload stego image for extraction

    if uploaded and user_key_input:  # Proceed only if image and key are provided
        try:
            user_key_bytes = user_key_input.encode()  # Encode entered key to bytes
            user_cipher = Fernet(user_key_bytes)  # Create Fernet cipher object with user key

            pil = Image.open(uploaded).convert("RGB")  # Open uploaded image and convert to RGB
            img_np = pil_to_np(pil)  # Convert to NumPy array (C, H, W)

            header_bits = extract_bits_from_image(img_np, 32)  # Extract first 32 bits for header
            length = int("".join(str(b) for b in header_bits), 2)  # Convert header bits to encrypted message length
            total_bits = 32 + length * 8  # Calculate total bits to extract (header + message)
            all_bits = extract_bits_from_image(img_np, total_bits)  # Extract all bits from image

            recovered = bits_to_text_with_custom_key(all_bits, user_cipher)  # Decrypt and decode message
            st.success("💬 Extracted Message:")  # Show success message
            st.text_area("Recovered message", recovered, height=150)  # Show recovered message in text area
        except Exception as e:
            st.error(f"❌ Error extracting message: {e}")  # Show error if extraction fails
    elif uploaded and not user_key_input:
        st.warning("⚠️ Please enter the encryption key to decrypt the message.")  # Warn user to enter key
