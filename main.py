import io
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_image(uploaded_file):
    if uploaded_file is None:
        return None
    else:
        image_data = uploaded_file.getvalue()

        st.image(image_data, width=200)

        st.write("Image Input : ", uploaded_file.name)
        
        return Image.open(io.BytesIO(image_data)).convert('RGB')


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")


@st.cache_resource
def load_model():
    return BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


model = load_model()

st.title("Project - Image to Text")
st.write("""
         #### TEAM MEMBER
         - Рахарди Сандикха РИМ-130908
         - Мухин Виктор Александрович РИМ-130908
         """)

st.write("""#### Our Project""")

uploadedFile = st.file_uploader('Upload image here')
raw_image = load_image(uploadedFile)

result = st.button('Submit')

if result:
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=1000)
    text_output = processor.decode(out[0], skip_special_tokens=True)
    st.write("=========================================")
    st.write("Output : ", str(text_output).capitalize())
