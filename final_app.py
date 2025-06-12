import streamlit as st  # Streamlit for web app UI
import numpy as np  # NumPy for numerical array manipulation
from PIL import Image  # PIL (Pillow) for image loading and saving
import pickle  # For loading CIFAR-10 dataset batches
import io  # For in-memory byte stream handling
import zlib  # For compressing and decompressing message data
import hashlib  # For SHA-256 hashing (integrity check)
from cryptography.fernet import Fernet  # For symmetric encryption/decryption (Fernet cipher)

st.set_page_config(layout="wide")  # Set the Streamlit page layout to wide screen in the begining to avoid bugs

# Encryption Key UI and management
st.sidebar.title("Encryption Key")  # Set sidebar title

if "fernet_key" not in st.session_state:  # Check if encryption key already exists in session state
    st.session_state.fernet_key = Fernet.generate_key()  # Generate and store a new Fernet encryption key

cipher = Fernet(st.session_state.fernet_key)  # Create cipher object using the stored key

st.sidebar.code(st.session_state.fernet_key.decode(), language="text")  # Display the current encryption key in sidebar

# CIFAR-10 dataset loading with caching to avoid reloading
@st.cache_data  # Cache the result of the function
def load_cifar10_images():
    with open('data_batch_1', 'rb') as fo:  # Open CIFAR-10 data file in binary read mode
        batch = pickle.load(fo, encoding='bytes')  # Load CIFAR-10 batch with byte encoding
    data = batch[b'data']  # Extract image data from batch
    return data.reshape(-1, 3, 32, 32).astype(np.uint8)  # Reshape to (num_images, channels, height, width) and convert to uint8

# Convert text to secure bit stream (compressed, hashed, encrypted, header)
def text_to_bits_with_header_secure(text):
    compressed = zlib.compress(text.encode('utf-8'))  # Compress UTF-8 encoded string using zlib
    hashed = hashlib.sha256(compressed).digest()  # Generate SHA-256 hash of compressed data
    combined = compressed + hashed  # Combine compressed data and hash (checksum)
    encrypted = cipher.encrypt(combined)  # Encrypt combined data with Fernet cipher
    header = [int(bit) for bit in format(len(encrypted), '032b')]  # Convert encrypted length to 32-bit header
    body = [int(bit) for byte in encrypted for bit in format(byte, '08b')]  # Convert encrypted bytes to bits
    return header + body  # Return final bit stream: header + encrypted message bits

# Convert extracted bits back to original text using custom cipher
def bits_to_text_with_custom_key(bits, cipher):
    header = bits[:32]  # Extract first 32 bits as header
    length = int(''.join(str(b) for b in header), 2)  # Convert header to integer (length of encrypted message)
    body = bits[32:32 + length * 8]  # Extract encrypted message bits
    encrypted_bytes = bytes([int(''.join(str(bit) for bit in body[i:i+8]), 2) for i in range(0, len(body), 8)])  # Convert bits to bytes
    try:
        combined = cipher.decrypt(encrypted_bytes)  # Decrypt encrypted data
        compressed = combined[:-32]  # Separate compressed message from checksum
        checksum = combined[-32:]  # Extract checksum (last 32 bytes)
        if hashlib.sha256(compressed).digest() != checksum:  # Validate checksum
            return "Integrity check failed."  # Return error if checksum does not match
        return zlib.decompress(compressed).decode('utf-8')  # Decompress and decode original message
    except:
        return "Decryption failed."  # Return error if decryption fails

# Embed bits into the image using LSB (Least Significant Bit) steganography
def embed_message_in_image(img_np, bits):
    flat = img_np.flatten()  # Flatten image array to 1D
    if len(bits) > len(flat):  # Check if message fits into image
        raise ValueError("Message too large for image.")  # Raise error if too big
    flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits  # Set LSB of image pixels to message bits
    return flat.reshape(img_np.shape)  # Reshape back to original image shape

# Extract LSB bits from the image
def extract_bits_from_image(img_np, num_bits):
    flat = img_np.flatten()  # Flatten image to 1D
    return (flat[:num_bits] & 1).tolist()  # Return LSB bits as list

# Convert PIL Image to NumPy array with shape (channels, height, width)
def pil_to_np(pil_img):
    pil_img = pil_img.convert("RGB")  # Ensure RGB format
    return np.array(pil_img).astype(np.uint8).transpose(2, 0, 1)  # Convert and transpose axes

# Convert NumPy image array back to PIL Image
def np_to_pil(img_np):
    return Image.fromarray(np.transpose(img_np, (1, 2, 0)).astype(np.uint8))  # Transpose and convert to PIL Image

# Streamlit UI setup
st.title("\U0001F510 Secure Image Steganography")  # Main title with lock emoji
tab1, tab2 = st.tabs(["\U0001F5BC\uFE0F Embed Message", "\U0001F50D Extract Message"])  # Two tabs for embed and extract

# Tab 1: Embed Message
with tab1:
    st.header("\U0001F510 Embed a Secret Message")  # Header title
    source = st.radio("Choose image source:", ["Random CIFAR-10 Image", "Upload Your Own"])  # Radio button for image source selection
    original_img_np = None  # Initialize image variable

    if source == "Random CIFAR-10 Image":
        images = load_cifar10_images()  # Load CIFAR-10 images
        idx = st.slider("Select CIFAR-10 Image Index", 0, len(images) - 1, 0)  # Slider to select image index
        original_img_np = images[idx]  # Select image by index
        st.image(np.transpose(original_img_np, (1, 2, 0)), caption=f"CIFAR-10 Image #{idx}", width=200)  # Show image
    else:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])  # File uploader for custom image
        st.caption("**Upload a PNG file (max ~20MB recommended due to platform limits)**")

        if uploaded:
            pil = Image.open(uploaded)  # Open image
            original_img_np = pil_to_np(pil)  # Convert to NumPy
            st.image(pil, caption="Uploaded High-Res Image", use_container_width=True)  # Display uploaded image

    if original_img_np is not None:  # If image is selected
        capacity_bits = original_img_np.size  # Total available bits in image
        max_payload_bits = capacity_bits - 32  # Reserve 32 bits for header
        max_payload_bytes = max_payload_bits // 8  # Convert bits to bytes
        est_encryption_overhead = 1.3  # Estimate encryption overhead (130%)
        max_input_bytes = int(max_payload_bytes / est_encryption_overhead)  # Adjust for overhead
        max_input_chars = int(max_input_bytes * 0.9)  # Estimate input character limit

        message = st.text_input(f"\U0001F4AC Secret message (max ~{max_input_chars} characters):", "")  # Message input field

        if message:
            input_bytes = len(message.encode('utf-8'))  # Calculate size of message
            st.caption(f"\U0001F50E Message size: {input_bytes} bytes (max ~{max_input_bytes} after encryption)")  # Display message size

            if input_bytes > max_input_bytes:
                st.warning("\u26A0\uFE0F Message too long to embed in this image after compression and encryption.")  # Warn if message too long
            else:
                if st.button("\U0001F510 Embed Message"):  # Embed button
                    try:
                        bits = text_to_bits_with_header_secure(message)  # Convert message to secure bits
                        stego_np = embed_message_in_image(original_img_np.copy(), bits)  # Embed in image
                        stego_pil = np_to_pil(stego_np)  # Convert back to PIL image
                        st.subheader("\U0001F50D Stego Image")  # Subheader for output image
                        st.image(stego_pil, caption="Image with Embedded Message", use_container_width=True)  # Show stego image
                        buf = io.BytesIO()  # Create in-memory buffer
                        stego_pil.save(buf, format="PNG")  # Save image to buffer
                        byte_img = buf.getvalue()  # Get byte data
                        st.download_button("\u2B07\uFE0F Download Stego Image", data=byte_img, file_name="stego_image.png", mime="image/png")  # Download button
                    except Exception as e:
                        st.error(f"\u274C Error embedding message: {e}")  # Display error if embedding fails

# Tab 2: Extract Message
with tab2:
    st.header("\U0001F50D Extract Hidden Message")  # Header for extract tab
    user_key_input = st.text_input("\U0001F511 Enter the encryption key used during embedding:")  # Key input for decryption
    uploaded = st.file_uploader("Upload image with hidden message", type=["png"], key="extract")  # Upload stego image
    st.caption("**Upload a PNG file (max ~20MB recommended due to platform limits)**")

    if uploaded and user_key_input:  # If both image and key provided
        try:
            user_key_bytes = user_key_input.encode()  # Convert input key to bytes
            user_cipher = Fernet(user_key_bytes)  # Create Fernet cipher with user key
            pil = Image.open(uploaded).convert("RGB")  # Open uploaded image and convert to RGB
            img_np = pil_to_np(pil)  # Convert to NumPy
            header_bits = extract_bits_from_image(img_np, 32)  # Extract header
            length = int("".join(str(b) for b in header_bits), 2)  # Decode header to get message length
            total_bits = 32 + length * 8  # Calculate total bits
            all_bits = extract_bits_from_image(img_np, total_bits)  # Extract full message bits
            recovered = bits_to_text_with_custom_key(all_bits, user_cipher)  # Decrypt and recover text
            st.success("\U0001F4AC Extracted Message:")  # Show success message
            st.text_area("Recovered message", recovered, height=150)  # Display recovered message
        except Exception as e:
            st.error(f"\u274C Error extracting message: {e}")  # Show extraction error
    elif uploaded and not user_key_input:
        st.warning("\u26A0\uFE0F Please enter the encryption key to decrypt the message.")  # Prompt for key if missing
