import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from neuron import NN

st.title("Brainz")
st.write("Digit guesser using Neural Networks")

st.info("""
**Instruction:**\n
    Draw BIG, BOLD and CENTERED single digit (0-9).""")

st.warning("""**N.B.** This is a simple neural network model with less epochs, so it is not perfect and may misclassify digits.""")

@st.cache_resource

def load_model():
    model = NN(layer_size=[784, 100, 10])
    try:
        model.load_weights("mnist_weights.npz")
        return model
    except:
        print("Failed to load model.")
        return None
    
neural = load_model()

if neural is None:
    st.error("No Brain No Gain!")

canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Guess"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:,:,0]

        sm_img = img[::10, ::10]
        sm_img = sm_img/255.0

        inputs = sm_img.reshape(784)

        pred = neural.forward(inputs)
        probs = pred.reshape(10)

        prob_digit = np.argsort(probs)[-3:][::-1]

        confirmed_digit = prob_digit[0]
        confirmed_confidence = probs[confirmed_digit]

        digit2 = prob_digit[1]
        confidence2 = probs[digit2]

        digit3 = prob_digit[2]
        confidence3 = probs[digit3]

        st.success(f"Guessed Digit: {confirmed_digit} with {confirmed_confidence*100:.2f}% confidence.")

        col1, col2 = st.columns(2)

        with col1:
            st.write("What the model sees:")
            st.image(sm_img, width=140)
        
        with col2:
            st.write("Other possibilities:")
            st.write(f"1. {digit2} with confidence {confidence2*100:.2f}%")
            st.write(f"2. {digit3} with confidence {confidence3*100:.2f}%")
