import os
import spaces
import gradio as gr
import torch
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import faiss  # FAISS for fast retrieval
import numpy as np

# Initialize FAISS index for fast similarity search (used only if selected)
embedding_dim = 448
faiss_index = faiss.IndexFlatL2(embedding_dim)
stored_images = []  # To store images associated with embeddings for retrieval if using FAISS


def preprocess_image(image_path, grayscale=False):
    """Apply optional grayscale and other enhancements to images."""
    img = Image.open(image_path)
    if grayscale:
        img = img.convert("L")  # Apply grayscale if selected
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  # Sharpen
    return img


@spaces.GPU
def model_inference(images, text, grayscale=False):
    """Qwen2VL-based inference function with optional grayscale processing."""
    images = [
        {
            "type": "image",
            "image": preprocess_image(image[0], grayscale=grayscale),
            "resized_height": 1344,
            "resized_width": 1344,
        }
        for image in images
    ]
    images.append({"type": "text", "text": text})

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to("cuda:0")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    messages = [{"role": "user", "content": images}]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text_input], images=image_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    del model, processor
    torch.cuda.empty_cache()
    return output_text[0]


@spaces.GPU
def search(query: str, ds, images, k, retrieval_method="CustomEvaluator"):
    """Search function with option to choose between CustomEvaluator and FAISS for retrieval."""
    model_name = "vidore/colpali-v1.2"
    token = os.environ.get("HF_TOKEN")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda", token=token
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_name, token=token)
    mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

    # Process the query to obtain embeddings
    batch_query = process_queries(processor, [query], mock_image)
    embeddings_query = model(**{k: v.to(device) for k, v in batch_query.items()})
    query_embedding = embeddings_query[0].cpu().numpy()

    if retrieval_method == "FAISS":
        # Use FAISS for efficient retrieval
        distances, indices = faiss_index.search(np.array([query_embedding]), k)
        results = [stored_images[idx] for idx in indices[0]]
    else:
        # Use CustomEvaluator for retrieval
        qs = [query_embedding]
        retriever_evaluator = CustomEvaluator(is_multi_vector=True)
        scores = retriever_evaluator.evaluate(qs, ds)

        top_k_indices = scores.argsort(axis=1)[0][-k:][::-1]
        results = [images[idx] for idx in top_k_indices]

    del model, processor
    torch.cuda.empty_cache()
    return results


def index(files, ds):
    """Convert and index PDF files."""
    images = convert_files(files)
    return index_gpu(images, ds)


def convert_files(files):
    """Convert PDF files to images."""
    images = []
    for f in files:
        images.extend(convert_from_path(f, thread_count=4))

    if len(images) >= 150:
        raise gr.Error("The number of images in the dataset should be less than 150.")
    return images


@spaces.GPU
def index_gpu(images, ds):
    """Index documents using FAISS or store in dataset for CustomEvaluator."""
    global stored_images
    model_name = "vidore/colpali-v1.2"
    token = os.environ.get("HF_TOKEN")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda", token=token
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_name, token=token)
    mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
    
    dataloader = DataLoader(images, batch_size=4, shuffle=False, collate_fn=lambda x: process_images(processor, x))
    all_embeddings = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings_doc = model(**batch)
            all_embeddings.extend(embeddings_doc.cpu().numpy())
    
    # Store embeddings in FAISS index and dataset for respective retrieval options
    embeddings = np.array(all_embeddings)
    faiss_index.add(embeddings)  # Add to FAISS index
    ds.extend(list(torch.unbind(torch.tensor(embeddings))))  # Extend original ds for CustomEvaluator
    stored_images.extend(images)  # Store images to link with FAISS indices

    del model, processor
    torch.cuda.empty_cache()
    return f"Indexed {len(images)} pages"


def get_example():
    return [
        [["RAPPORT_DEVELOPPEMENT_DURABLE_2019.pdf"], "Quels sont les 4 axes majeurs des achats?"],
        [["RAPPORT_DEVELOPPEMENT_DURABLE_2019.pdf"], "Quelles sont les actions entreprise en Afrique du Sud?"],
        [["RAPPORT_DEVELOPPEMENT_DURABLE_2019.pdf"], "fais moi un tableau markdown de la répartition homme femme"],
    ]


with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 📝 ColPali + Qwen2VL 7B: Enhanced Document Retrieval & Analysis App")

    # Section 1: File Upload
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Step 1: Upload Your Documents 📄")
            file = gr.File(file_types=["pdf"], file_count="multiple", label="Upload PDF Documents")
            grayscale_option = gr.Checkbox(label="Convert images to grayscale 🖤", value=False)
            convert_button = gr.Button("🔄 Index Documents", variant="secondary")
            message = gr.Textbox("No files uploaded yet", label="Status", interactive=False)
            embeds = gr.State(value=[])
            imgs = gr.State(value=[])
            img_chunk = gr.State(value=[])

    # Section 2: Search Options
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Step 2: Search the Indexed Documents 🔍")
            query = gr.Textbox(placeholder="Enter your query here", label="Query", lines=2)
            k = gr.Slider(minimum=1, maximum=10, step=1, label="Number of Results", value=1)
            retrieval_method = gr.Dropdown(
                choices=["CustomEvaluator", "FAISS"], 
                label="Choose Retrieval Method 🔀",
                value="CustomEvaluator"
            )
            search_button = gr.Button("🔍 Search", variant="primary")
    
    # Displaying Examples
    with gr.Row():
        gr.Markdown("## 💡 Example Queries")
        gr.Examples(examples=get_example(), inputs=[file, query], label="Try These Examples", show_label=True)

    # Output Gallery for Search Results
    output_gallery = gr.Gallery(label="📂 Retrieved Documents", height=600, show_label=True)

    # Section 3: Answer Retrieval
    with gr.Row():
        gr.Markdown("## Step 3: Generate Answers with Qwen2-VL 🧠")
        answer_button = gr.Button("💬 Get Answer", variant="primary")
        output = gr.Markdown(label="Output")

    # Define interactions
    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])
    search_button.click(search, inputs=[query, embeds, imgs, k, retrieval_method], outputs=[output_gallery])
    answer_button.click(model_inference, inputs=[output_gallery, query, grayscale_option], outputs=output)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(share=True)
