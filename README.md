# Multi-modal RAG pipeline

This is a working example of a multi-modal RAG QA application that lets users upload a PDF file and asks about the content of the file. The system return a textual answer with relevant images if available.

<b>Author:</b> Jan Čuhel

<b>Date:</b>May 2024

## Architecture

### Pre-processing

![Pre-processing phase of the multi-modal RAG pipeline](./images/multimodal_plot_rag_preprocessing.png)

### Inference

![Inference phase of the multi-modal RAG pipeline](./images/multimodal_inference_rag.png)

## Installation

```bash
# Activate a python environment of your choice (e.g. venv, Conda)
# ...
# Install the dependencies
pip intall -r requirements.txt
```

## Execution

```bash
python multimodal_rag_pipeline.py --source_file manual.pdf
```

## Hardware requirements

We recommend to run this application on a device with a strong GPU as it utilizes several Deep Learning Models.

## Intended use

The intended use of the system is for vehicle manuals, but it can be used for different manuals as well.
