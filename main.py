import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
import nltk
from optimum.intel import OVModelForCausalLM
import faiss
import gc
import gradio as gr
import tempfile
import torch
from chatbot_state import ChatbotState
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

nltk.download('punkt')
app = Flask(__name__)

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text



def create_semantic_chunks(text, chunk_size=1000, overlap=200):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap:] + sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def create_vector_store(text, chunk_size=1000, overlap=200):
    chunks = create_semantic_chunks(text, chunk_size, overlap)
    em_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = em_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks, em_model


model_id = "path to model"
tokenizer= AutoTokenizer.from_pretrained(model_id)
chat_model = OVModelForCausalLM.from_pretrained(model_id, device="GPU")

def setup_chatbot(pdf_file, model, tokenizer):
    # Read the PDF file directly from the file-like object
    text = read_pdf(pdf_file)
    
    index, chunks, embedding_model = create_vector_store(text)
    return index, chunks, embedding_model, model, tokenizer
# Function to create semantic chunks from text





def chatbot(query, index, chunks, embedding_model, llm_model, tokenizer, k=3, max_input_length=1024, max_new_tokens=170, similarity_threshold=0.3):

    # Find relevant chunks
    query_vector = embedding_model.encode([query])
    print("thiss")
    _, I = index.search(query_vector, k)
    relevant_chunks = [chunks[i] for i in I[0]]
    print("no thissss")
    # Calculate similarity between the query and the retrieved chunks
    chunk_vectors = embedding_model.encode(relevant_chunks)
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    
    # Check if any chunk is sufficiently similar to the query
    if np.max(similarities) < similarity_threshold:
        return "I can't answer that question. That information is not in this document."

    # Construct Prompt
    context = "\n".join(relevant_chunks)
    prompt = (
        f"Use the following context information to answer the user's question based solely on the provided context. "
        f"Do not include any additional questions or information not found in the context.\n\n"
        f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    )

    # Generate Response
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    output = llm_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=False,
        top_p=0.9,
        num_beams=5,
        early_stopping=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract Answer
    if "Answer:" in response:
        answer_start = response.find("Answer:") + len("Answer:")
        
        answer_end = response.find("Question:", answer_start)
        if answer_end == -1:  # No "Question:" found, take till end
            answer_end = len(response)
        answer = response[answer_start:answer_end].strip()
    else:
        answer = "I don't know."

    return answer




chatbot_state = ChatbotState()




UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        chatbot_state.index, chatbot_state.chunks, chatbot_state.embedding_model, chatbot_state.llm_model, chatbot_state.tokenizer = setup_chatbot(file, chat_model, tokenizer)
        chatbot_state.pdf_uploaded = True
        return jsonify({"message": "PDF uploaded and processed successfully."}), 200
    else:
        return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route('/ask_question', methods=['POST'])
def ask_question():
    if not chatbot_state.pdf_uploaded:
        return jsonify({'error': 'Please upload a PDF file first.'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if question:
        try:
            answer = chatbot(question, chatbot_state.index, chatbot_state.chunks, chatbot_state.embedding_model, chatbot_state.llm_model, chatbot_state.tokenizer)
            return jsonify({'answer': answer}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No question provided'}), 400

@app.route('/restart', methods=['POST'])
def restart():
    chatbot_state.index = None
    chatbot_state.chunks = None
    chatbot_state.embedding_model = None
    chatbot_state.llm_model = None
    chatbot_state.tokenizer = None
    chatbot_state.pdf_uploaded = False

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return jsonify({'message': 'Chatbot restarted. Please upload a new PDF.'}), 200

if __name__ == '__main__':
    app.run(debug=True)
