import os
import fitz
from PIL import Image
import io
import Levenshtein
from typing import List, Tuple
import torch
from tqdm import tqdm
from transformers import (
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration, 
    Pix2StructProcessor, 
    Pix2StructForConditionalGeneration,
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration
)
from langchain_core.documents import Document


# Constants
BBOX_WIDTH = 40
SINGLE_BBOX_WIDTH = 30
MIN_TEXT_LENGTH = 2
HISTORY_LOOKING_LENGTH = 6


def find_matches_with_edit_distance(text: str, template: str, threshold: int=2) -> List[Tuple[int, int, int]]:
    """
    Find matches with edit distance

    Args:
        text (str): The text to search in
        template (str): The template to search for
        threshold (int, optional): The maximum edit distance. Defaults to 2.

    Returns:
        List[Tuple[int, int, int]]: The list of matches with edit distance
    """
    matches = []

    template_length = len(template)
    text_length = len(text)

    for i in range(text_length - template_length + 1):
        substring = text[i:i + template_length]
        distance = Levenshtein.distance(substring, template)

        if distance <= threshold:
            matches.append((i, i + template_length - 1, distance))

    # Sort matches based on edit distance
    matches.sort(key=lambda x: (x[2], x[0], x[1]))

    return matches


def find_match_around(text: str, reverse_mode: bool=True) -> bool:
    """
    Find a match around the text

    Args:
        text (str): The text to search around
        reverse_mode (bool, optional): Whether to reverse the text. Defaults to True.

    Returns:
        bool: Whether a match was found
    """
    if reverse_mode:
        text = text[::-1]
    for ch in text:
        if ch == ' ':
            continue
        elif ch == "\t" or ch == '\n':
            return True
        elif ch.isalnum():
            return False
    return False


class MultiModalPDFLoader:
    def __init__(self, file_path: str, min_height: int=100, min_width: int=100, extracted_images_path: str="extracted_images", use_plots: bool=False):
        """
        Initialize the loader
        
        Args:
            file_path (str): The file path
        """
        self.file_path = file_path
        # Check if the source file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file {file_path} not found.")
        # Check if the source file is a PDF
        if not file_path.endswith(".pdf"):
            raise ValueError(f"Source file {file_path} is not a PDF.")
        self.min_height = min_height
        self.min_width = min_width
        self.extracted_images_path = extracted_images_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.icon_captioner_processor, self.icon_captioner_model = self.initialize_icon_captioner()
        self.create_image_folder()
        self.use_plots = use_plots
        if self.use_plots:
            self.plot_processor, self.plot_model = self.initialize_plot_translator()
            self.plot_recognizer_processor, self.plot_recognizer_model = self.initialize_plot_recognizer()
        
    def create_image_folder(self):
        """
        Create the image folder
        """
        if not os.path.exists(self.extracted_images_path):
            os.makedirs(self.extracted_images_path)
    
    def initialize_icon_captioner(self):
        """
        Initialize the icon captioner
        """
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl", 
            device_map=self.device, 
            torch_dtype=torch.float16 if self.device == "cuda" else "auto", 
            low_cpu_mem_usage=True
        )
        return processor, model
    
    def caption_image(self, img: Image.Image) -> str:
        """
        Caption the image
        
        Args:
            img (Image.Image): The image
            
        Returns:
            str: The caption
        """
        img = img.convert("RGB") if img.mode != "RGB" else img

        prompt = "Generate a caption for this given icon from a manual of an electric vehicle."
        inputs = self.icon_captioner_processor(images=img, text=prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.icon_captioner_model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_new_tokens=100,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            ).to("cpu")
        return self.icon_captioner_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    def initialize_plot_translator(self):
        """
        Initialize the plot translator
        """
        processor = Pix2StructProcessor.from_pretrained('google/deplot')
        model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(self.device)
        return processor, model

    def translate_plot(self, img: Image.Image) -> str:
        """
        Translate the plot
        
        Args:
            img (Image.Image): The image
            
        Returns:
            str: The translation
        """
        img = img.convert("RGB") if img.mode != "RGB" else img

        prompt = "Generate underlying data table of the figure below:"
        inputs = self.plot_processor(images=img, text=prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.plot_model.generate(**inputs, max_new_tokens=512,).to("cpu")
        
        return self.plot_processor.decode(outputs[0], skip_special_tokens=True).strip()

    def initialize_plot_recognizer(self):
        """
        Initialize the plot recognizer
        """
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype="auto", device_map="auto")
        return processor, model

    def is_plot(self, img: Image.Image) -> bool:
        """
        Check if the image is a plot
        
        Args:
            img (Image.Image): The image
            
        Returns:
            bool: Whether the image is a plot
        """
        img = img.convert("RGB") if img.mode != "RGB" else img

        prompt = "[INST] <image>\nDecide whether this image is a chart, plot or something different (e.g. a diagram or flowchart). In case this image is either plot or chart, reply 'Yes', if the image is a diagram, flowchart or something else reply 'No'. Do not reply anything else. [/INST]"
        inputs = self.plot_recognizer_processor(prompt, img, return_tensors="pt")
        if self.device != "auto":
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.plot_recognizer_model.generate(
                **inputs, 
                max_new_tokens=5,
                pad_token_id=self.plot_recognizer_processor.tokenizer.eos_token_id
            ).to("cpu")
        answer = self.plot_recognizer_processor.decode(outputs[0], skip_special_tokens=True).strip()
        answer = answer[answer.index("[/INST]")+7:]

        return True if answer.lower() == "yes" or "yes" in answer.lower() else False
    
    def load(self, debug=False, max_pages=None):
        """
        Load the PDF
        """
        pdf_file = fitz.open(self.file_path)
        
        docs = []

        icon_idx = 0
        icon_total_locations = 0

        # for page_index in tqdm(range(len(pdf_file)), desc="Processing pages"):
        n = min(len(pdf_file), max_pages) if max_pages is not None else len(pdf_file)
        for page_index in tqdm(range(n), desc="Processing pages"):
            # get the page itself
            page = pdf_file[page_index]
            image_list = page.get_images()
            text_page = page.get_text()
            image_paths = []

            # printing number of images found in this page
            if image_list and debug:
                print(
                    f"[+] Found a total of {len(image_list)} images in page {page_index+1}")

            for image_index, img in enumerate(page.get_images(full=True), start=1):
                # get the XREF of the image
                xref = img[0]

                # extract the image bytes
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]

                # Load it to PIL
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Icon
                if image.width <= self.min_width and image.height <= self.min_height:
                    # Get the caption
                    icon_caption = self.caption_image(image)
                    # Get all bboxes
                    bboxes = page.get_image_rects(img)
                    icon_total_locations += len(bboxes)
                    # Index for splitting the text
                    cut_idx = 0
                    # For each bbox
                    for bbox in bboxes:
                        rect = fitz.Rect(max(0, bbox.x0-SINGLE_BBOX_WIDTH), bbox.y0, bbox.x1+SINGLE_BBOX_WIDTH, bbox.y0+5)
                        text_around = page.get_textbox(rect).strip()
                        rect_left = fitz.Rect(max(0, bbox.x0-SINGLE_BBOX_WIDTH), bbox.y0, bbox.x0, bbox.y0+5)
                        text_left = page.get_textbox(rect_left).strip()
                        rect_right = fitz.Rect(bbox.x1, bbox.y0, bbox.x1+SINGLE_BBOX_WIDTH, bbox.y0+5)
                        text_right = page.get_textbox(rect_right).strip()

                        matches = []
                        if len(text_around) > MIN_TEXT_LENGTH:
                            matches += find_matches_with_edit_distance(text_page[cut_idx:], text_around, threshold=min(round(len(text_around)/3), 8))
                        matches_left = []
                        if len(text_left) > MIN_TEXT_LENGTH:
                            matches_left += find_matches_with_edit_distance(text_page[cut_idx:], text_left, threshold=min(round(len(text_left)/3), 7))
                        matches_right = []
                        if len(text_right) > MIN_TEXT_LENGTH:
                            matches_right += find_matches_with_edit_distance(text_page[cut_idx:], text_right, threshold=min(round(len(text_right)/3), 7))

                        if debug:
                            caption_left = f"Surrounding text left: '{text_left}' (box: {rect_left})"
                            caption_right = f"Surrounding text right: '{text_right}' (box: {rect_right})"
                            caption_around = f"Surrounding text around: '{text_around}' (box: {rect})"
                            print("-"*30)
                            print(f"[-] Image {image_index} on page {page_index+1} with width={image.width} and height={image.height} has an around caption={caption_around}, a left caption={caption_left} and right caption={caption_right}")
                            print(f"[-] Coordinates x0: {bbox.x0} x1: {bbox.x1} y0: {bbox.y0} y1: {bbox.y1}")
                        move_delimiter = 0

                        if len(matches) > 0:
                            solution_idx = -1
                            # Sort matches based on edit distance
                            if len(text_left) > 0 and len(text_right)> 0:
                                solution_idx = 0
                            # There is no text on the left
                            elif len(text_left) == 0:
                                # Iterate though the matches
                                for match_idx, (start_idx, end_idx, distance) in enumerate(matches):
                                    str_before = text_page[cut_idx+start_idx-HISTORY_LOOKING_LENGTH:cut_idx+start_idx]
                                    # print(f'---- {match_idx}: {str_before} ({start_idx}:{end_idx}:{distance})')
                                    if find_match_around(str_before):
                                        # Find solution
                                        solution_idx = match_idx
                                        move_delimiter = -1
                                        break
                            elif len(text_right) == 0:
                                # Iterate though the matches
                                for match_idx, (start_idx, end_idx, distance) in enumerate(matches):
                                    if find_match_around(text_page[cut_idx+start_idx-HISTORY_LOOKING_LENGTH:cut_idx+start_idx], reverse_mode=False):
                                        # Find solution
                                        solution_idx = match_idx
                                        break

                            # There is a solution
                            if solution_idx >= 0:
                                start_idx, end_idx, distance = matches[solution_idx]
                                if debug:
                                    print(f'[-] Found match with #ICON{icon_idx} distance: {distance} with {cut_idx+start_idx}:{cut_idx+end_idx} indices')
                                # delimiter_idx = cut_idx+end_idx+1
                                delimiter_idx = cut_idx+start_idx+len(text_left)+move_delimiter+1
                                text_page = text_page[:delimiter_idx] + f" #ICON{icon_idx}({icon_caption}) " + text_page[delimiter_idx:]
                                cut_idx += end_idx+1
                        if debug:
                            print("-"*30)
                    icon_idx += 1
                else:
                    # Check whether the image is a plot
                    if self.use_plots and self.is_plot(image):
                        if debug:
                            print(f"[-] Plot detected! Image {image_index} on page {page_index+1} with width={image.width} and height={image.height}")
                        # Translate the plot
                        plot_translation = self.translate_plot(image)
                        # Append to docs
                        docs.append(
                            Document(page_content=plot_translation, metadata={"source": self.file_path, "page": page_index+1, "images": []})
                        )
                    else:
                        # Save the image
                        page_folder = os.path.join(self.extracted_images_path, f"page{page_index+1}")
                        if not os.path.exists(page_folder):
                            os.makedirs(page_folder)
                        # Save the image
                        image_path = os.path.join(page_folder, f"image_{image_index}.png")
                        image.save(open(image_path, "wb"), format="png")
                        image_paths.append(image_path)
            # Update the page
            docs.append(
                Document(page_content=text_page, metadata={"source": self.file_path, "page": page_index+1, "images": image_paths})
            )

        return docs

    def release(self):
        """
        Clear the GPU
        """
        self.icon_captioner_model.cpu()
        if self.use_plots:
            self.plot_model.cpu()
            self.plot_recognizer_model.cpu()
        if self.device == "cuda":
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    loader = MultiModalPDFLoader("BMW_i4.pdf", use_plots=True)
    data = loader.load(debug=False, max_pages=100)
    # print(data[6])
    # print(len(data))
