import gradio as gr

from src.predict import (
    load_trained_model,
    predict_single_image
)

model = load_trained_model()

def predict(image):

    prediction, confidence, _ = (
        predict_single_image(
            model,
            image
        )
    )

    return f"{prediction} ({confidence:.4f})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="CIFAR10 Classifier"
)

demo.launch()