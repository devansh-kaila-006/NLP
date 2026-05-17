"""
Rename MIT 6.S191 DL Alternative Transcripts to Systematic Names
"""

import os
from pathlib import Path

# File mappings: original NoteGPT name -> systematic name
FILE_MAPPINGS = {
    "NoteGPT_TRANSCRIPT_1 Introduction to Neural Networks and Deep Learning; Training Deep NNs.srt":
        "MIT_DL_L01_I_Introduction_to_Neural_Networks_Training_Deep_NNs.srt",

    "NoteGPT_TRANSCRIPT_2 Training Deep NNs (cont.); Introduction to KerasTensorflow; Application to Tabular Data.srt":
        "MIT_DL_L02_I_Training_Deep_NNs_Cont_Introduction_to_KerasTensorflow_Application_to_Tabular_Data.srt",

    "NoteGPT_TRANSCRIPT_3 Deep Learning for Computer Vision – Building Convolutional Neural Networks from Scratch.srt":
        "MIT_DL_L03_I_Deep_Learning_for_Computer_Vision_Building_CNNs_from_Scratch.srt",

    "NoteGPT_TRANSCRIPT_4 Deep Learning for Computer Vision – Transfer Learning and Fine-Tuning; Intro to HuggingFace.srt":
        "MIT_DL_L04_I_Deep_Learning_for_Computer_Vision_Transfer_Learning_Intro_to_HuggingFace.srt",

    "NoteGPT_TRANSCRIPT_5 Deep Learning for Natural Language – The Basics.srt":
        "MIT_DL_L05_I_Deep_Learning_for_Natural_Language_The_Basics.srt",

    "NoteGPT_TRANSCRIPT_6 Deep Learning for Natural Language – Embeddings.srt":
        "MIT_DL_L06_I_Deep_Learning_for_Natural_Language_Embeddings.srt",

    "NoteGPT_TRANSCRIPT_7 Deep Learning for Natural Language – Transformers.srt":
        "MIT_DL_L07_I_Deep_Learning_for_Natural_Language_Transformers.srt",

    "NoteGPT_TRANSCRIPT_8 Deep Learning for Natural Language – Transformers, Self-Supervised Learning.srt":
        "MIT_DL_L08_I_Deep_Learning_for_Natural_Language_Transformers_Self_Supervised_Learning.srt",

    "NoteGPT_TRANSCRIPT_9 Generative AI – Large Language Models (LLMs) and Retrieval Augmented Generation (RAG).srt":
        "MIT_DL_L09_I_Generative_AI_LLMs_and_Retrieval_Augmented_Generation_RAG.srt",

    "NoteGPT_TRANSCRIPT_10 Generative AI – Adapting LLMs with Parameter-Efficient Fine-Tuning.srt":
        "MIT_DL_L10_I_Generative_AI_Adapting_LLMs_with_Parameter_Efficient_Fine_Tuning.srt",

    "NoteGPT_TRANSCRIPT_11 Generative AI – Text-to-Image Models.srt":
        "MIT_DL_L11_I_Generative_AI_Text_to_Image_Models.srt"
}

def rename_mit_dl_transcripts(transcript_dir):
    """Rename MIT DL transcripts to systematic names"""

    transcript_path = Path(transcript_dir)

    if not transcript_path.exists():
        print(f"Error: Directory not found: {transcript_dir}")
        return

    print("="*60)
    print("RENAMING MIT 6.S191 DL ALTERNATIVE TRANSCRIPTS")
    print("="*60)

    renamed_count = 0
    failed_count = 0

    for old_name, new_name in FILE_MAPPINGS.items():
        old_file = transcript_path / old_name
        new_file = transcript_path / new_name

        if old_file.exists():
            try:
                # Skip if new name already exists
                if new_file.exists():
                    print(f"[SKIP] {new_name} already exists")
                    # Delete old file
                    old_file.unlink()
                    renamed_count += 1
                    continue

                # Rename file
                old_file.rename(new_file)
                print(f"[OK] {old_name[:50]}... -> {new_name}")
                renamed_count += 1

            except Exception as e:
                print(f"[ERROR] Failed to rename {old_name}: {e}")
                failed_count += 1
        else:
            print(f"[NOT FOUND] {old_name}")

    print(f"\n{'='*60}")
    print(f"RENAME SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully renamed: {renamed_count}/{len(FILE_MAPPINGS)}")
    print(f"Failed: {failed_count}")

    # Create metadata file
    create_metadata(transcript_path)

    print(f"\n[SUCCESS] MIT DL transcripts renamed and organized!")
    print(f"Location: {transcript_path}")


def create_metadata(transcript_path):
    """Create metadata file for MIT DL transcripts"""

    metadata_content = """MIT 6.S191: Introduction to Deep Learning (Alternative Playlist)
========================================================================

Course Information:
- Course: MIT 6.S191 Introduction to Deep Learning
- Institution: Massachusetts Institute of Technology
- Playlist URL: https://www.youtube.com/playlist?list=PLUl4u3cNGP60YyhMjYmXuVmX562QcClSp
- Total Lectures: 11
- Total Duration: ~13.75 hours

Lecture List:
-------------
01. Introduction to Neural Networks and Deep Learning; Training Deep NNs
02. Training Deep NNs (cont.); Introduction to KerasTensorflow; Application to Tabular Data
03. Deep Learning for Computer Vision – Building Convolutional Neural Networks from Scratch
04. Deep Learning for Computer Vision – Transfer Learning and Fine-Tuning; Intro to HuggingFace
05. Deep Learning for Natural Language – The Basics
06. Deep Learning for Natural Language – Embeddings
07. Deep Learning for Natural Language – Transformers
08. Deep Learning for Natural Language – Transformers, Self-Supervised Learning
09. Generative AI – Large Language Models (LLMs) and Retrieval Augmented Generation (RAG)
10. Generative AI – Adapting LLMs with Parameter-Efficient Fine-Tuning
11. Generative AI – Text-to-Image Models

Topics Covered:
--------------
- Neural Networks and Deep Learning fundamentals
- Training Deep Neural Networks
- Keras/TensorFlow framework
- Computer Vision with CNNs
- Transfer Learning
- Natural Language Processing basics
- Word Embeddings
- Transformer architecture
- Self-Supervised Learning
- Large Language Models (LLMs)
- Retrieval Augmented Generation (RAG)
- Parameter-Efficient Fine-Tuning (PEFT)
- Text-to-Image Models

File Naming Convention:
----------------------
MIT_DL_LXX_T_Topic.srt

Where:
- XX: Lecture number (01-11)
- T: Type indicator (I = Lecture)
- Topic: Brief description of lecture content

Transcripts Source:
------------------
NoteGPT (30-second interval transcripts with timestamps)

Processing Status:
-----------------
[ ] Upload transcripts (completed)
[ ] Rename transcripts (completed)
[ ] Process video chunks
[ ] Generate embeddings
[ ] Create FAISS index
[ ] Test multi-modal queries

Last Updated: 2026-05-17
"""

    metadata_file = transcript_path / "MIT_DL_Metadata.txt"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        print(f"\n[OK] Created metadata file: MIT_DL_Metadata.txt")
    except Exception as e:
        print(f"[ERROR] Failed to create metadata file: {e}")


if __name__ == "__main__":
    # Rename MIT DL transcripts
    transcript_dir = "data/transcripts/dl_alt"

    rename_mit_dl_transcripts(transcript_dir)

    print("\nNext steps:")
    print("1. Run: python scripts/process_video_pipeline.py")
    print("2. Run: python scripts/create_video_index.py")
    print("3. Test: python -m src.pipeline.multimodal_rag_pipeline")
