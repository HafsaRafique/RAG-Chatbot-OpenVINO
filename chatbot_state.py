class ChatbotState:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.embedding_model = None
        self.llm_model = None
        self.tokenizer = None
        self.pdf_uploaded = False

    