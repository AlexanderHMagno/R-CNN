import gradio as gr
from inference import detect_planes
from PIL import Image, ImageDraw

def draw_boxes(image_path):
    prediction = detect_planes(image_path)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    print(prediction)

    for i in range(len(prediction[0]["boxes"])):
        box = prediction[0]["boxes"][i].cpu().numpy()
        score = prediction[0]["scores"][i].item()
        print(score)
        if score > 0.1:  # Confidence threshold
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1] - 10), f"Plane {score:.2f}", fill="red")

    return image

# Create Gradio UI
demo = gr.Interface(
    fn=draw_boxes,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(),
    title="Plane Detector",
    description="Upload an image, and the model will detect planes."
)

if __name__ == "__main__":
    demo.launch()
