import argparse
import os
import torch
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    pipeline, 
    AutoProcessor, 
    AutoModel
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple

from extractor import MultiModalPDFLoader


class MultimodalRAGPipeline:
    def __init__(
            self, 
            source_file: str,
            llm_generator: str = "mistralai/Mistral-7B-Instruct-v0.2",
            embeddings_model: str = "BAAI/bge-base-en-v1.5",
            use_plots: bool = True,
            max_pages: int = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            top_k: int = 3
        ):
        """
        Args:
            source_file (str): The path to the source PDF file.
            llm_generator (str): The name of the language model to use for text generation.
            embeddings_model (str): The name of the embeddings model to use for document retrieval.
            use_plots (bool): Whether to use plots for document retrieval.
            max_pages (int): The maximum number of pages to process from the source file.
            chiunk_size (int): The size of the chunks to split the documents into.
            chunk_overlap (int): The overlap between the chunks.
            top_k (int): The number of documents to retrieve.
        """
        self.llm_generator = llm_generator
        self.source_file = source_file
        self.embeddings_model = embeddings_model
        self.use_plots = use_plots
        self.max_pages = max_pages
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
        self.img_text_relevance_threshold = 0.1
        # Check if the source file exists
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file {source_file} not found.")
        # Check if the source file is a PDF
        if not source_file.endswith(".pdf"):
            raise ValueError(f"Source file {source_file} is not a PDF.")
        # Load the documents
        self._load_documents()
        # Split the documents
        self._split_documents()
        # Create the database
        self._create_db()
        # Create the RAG pipeline
        self._create_rag_pipeline()
        # Initialize the image-text relevance model
        self._initialize_img_text_relevance_model()
        
    def _load_documents(self):
        """
        Load the documents from the source file.
        """
        # Load the source file
        self.document_loader = MultiModalPDFLoader(self.source_file, use_plots=self.use_plots)
        # Load the documents
        self.docs = self.document_loader.load(debug=False, max_pages=self.max_pages)
        # Release
        self.document_loader.release()
    
    def _split_documents(self):
        """
        Split the documents into chunks for processing.
        """
        # Split the documents into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.documents = self.text_splitter.split_documents(self.docs)

    def _create_db(self):
        """
        Create the database for document retrieval.
        """
        # Create the database
        self.db = FAISS.from_documents(self.documents, HuggingFaceEmbeddings(model_name=self.embeddings_model))
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k})
    
    def _format_docs(self, docs) -> str:
        """
        Format the documents into a single string.

        Args:
            docs (list): The list of documents.

        Returns:
            str: The formatted documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_rag_pipeline(self):
        """
        Create the RAG pipeline for text generation.
        """
        # Create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_generator, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        # Create the model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_generator, quantization_config=bnb_config)
        # Create the text generation pipeline
        self.text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False,
            max_new_tokens=1000,
        )

        # Create the RAG pipeline
        self.llm = HuggingFacePipeline(pipeline=self.text_generation_pipeline)
        # Create the prompt template

        self.prompt_template = """
[INST]
You are an assistant for answering questions about electric vehicles.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Here is context to help:

{context}

### QUESTION:
{question} [/INST]
"""
        # Create the prompt from the prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template,
        )
        # Create the RAG chain
        self.rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: self._format_docs(x["context"])))
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=self.rag_chain_from_docs)

    def _initialize_img_text_relevance_model(self):
        """
        Initialize the image-text relevance model.
        """
        self.siglip_processor = AutoProcessor.from_pretrained("jancuhel/google-siglip-large-patch16-384-img-text-relevancy")
        self.siglip_model = AutoModel.from_pretrained("jancuhel/google-siglip-large-patch16-384-img-text-relevancy")
        self.siglip_model.to(self.device)

    def _get_image_text_relevance(self, text: str, img) -> bool:
        """
        Get the image-text relevance.

        Args:
            text (str): The text.
            img: The image.

        Returns:
            bool: Whether the image is relevant to the text.
        """
        inputs = self.siglip_processor(
            text=[text], 
            images=img.convert("RGB") if img.mode != "RGB" else img,
            padding="max_length",
            truncation=True, 
            return_tensors="pt"
        )
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.siglip_model(**inputs)

        logits_per_image = outputs.logits_per_image.detach().cpu()

        return torch.sigmoid(logits_per_image)[0].item() >= self.img_text_relevance_threshold
    
    def ask_question(self, question: str, save_images: bool = False, debug: bool = False) -> Tuple[str, List[Image.Image]]:
        """
        Ask a question and get the answer.

        Args:
            question (str): The question to ask.
            save_images (bool): Whether to save the relevant images.
            debug (bool): Whether to print debug information.

        Returns:
            str: The answer to the question.
            List[Image.Image]: The relevant images.
        """
        result = self.rag_chain_with_source.invoke(question)
        retrieved_images = []
        for doc in result["context"]:
            if len(doc.metadata["images"]) > 0:
                retrieved_images.extend(doc.metadata["images"])
        retrieved_images = list(set(retrieved_images))
        relevant_img_idx = 1
        relevant_images = []
        for img_path in retrieved_images:
            img = Image.open(img_path)
            if self._get_image_text_relevance(result["answer"], img):
                if debug:
                    print(f"Image: {img_path} is relevant to the text.")
                if save_images:
                    output_file = f"relevant_img_{relevant_img_idx}.png"
                    img.save(output_file)
                    if debug:
                        print(f"Image: {img_path} is saved to {output_file}.")
                relevant_img_idx += 1
                relevant_images.append(img)
        return result["answer"], relevant_images


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source_file", type=str, required=True, help="The path to the source PDF file.")
    args = argparser.parse_args()

    pipeline = MultimodalRAGPipeline(source_file=args.source_file, max_pages=50)
    print("Welcome to the Multimodal RAG Pipeline example!")
    print("Please ask a question about electric vehicle. To end the conversation, type 'exit'.")

    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        answer, _ = pipeline.ask_question(question)
        print(answer)


if __name__ ==  "__main__":
    main()
