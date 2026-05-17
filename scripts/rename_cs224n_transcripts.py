"""
Rename CS224n NLP Transcripts to Systematic Names
"""

import os
from pathlib import Path

# File mappings: original NoteGPT name -> systematic name
FILE_MAPPINGS = {
    # Spring 2024 Main Lectures (Lecture 1-18)
    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  Spring 2024  Lecture 1 - Intro and Word Vectors.srt":
        "CS224n_L01_I_Intro_and_Word_Vectors.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  Spring 2024  Lecture 2 - Word Vectors and Language Models.srt":
        "CS224n_L02_I_Word_Vectors_and_Language_Models.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  Spring 2024  Lecture 3 - Backpropagation, Neural Network.srt":
        "CS224n_L03_I_Backpropagation_Neural_Networks.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  Spring 2024  Lecture 4 - Dependency Parsing.srt":
        "CS224n_L04_I_Dependency_Parsing.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  Spring 2024  Lecture 5 - Recurrent Neural Networks.srt":
        "CS224n_L05_I_Recurrent_Neural_Networks.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  Spring 2024  Lecture 6 - Sequence to Sequence Models.srt":
        "CS224n_L06_I_Sequence_to_Sequence_Models.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  Spring 2024  Lecture 7 - Attention, Final Projects and LLM Intro.srt":
        "CS224n_L07_I_Attention_Final_Projects_LLM_Intro.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  Spring 2024  Lecture 10 - Post-training by Archit Sharma.srt":
        "CS224n_L10_I_Post_training_by_Archit_Sharma.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  Spring 2024  Lecture 11 - Benchmarking by Yann Dubois.srt":
        "CS224n_L11_I_Benchmarking_by_Yann_Dubois.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  Spring 2024  Lecture 12 - Efficient Training, Shikhar Murty.srt":
        "CS224n_L12_I_Efficient_Training_by_Shikhar_Murty.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL Spring 2024  Lecture 13 - Brain-Computer Interfaces, Chaofei Fan.srt":
        "CS224n_L13_I_Brain_Computer_Interfaces_by_Chaofei_Fan.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  Spring 2024  Lecture 14 - Reasoning and Agents by Shikhar Murty.srt":
        "CS224n_L14_I_Reasoning_and_Agents_by_Shikhar_Murty.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  Spring 2024  Lecture 15 - After DPO by Nathan Lambert.srt":
        "CS224n_L15_I_After_DPO_by_Nathan_Lambert.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  Spring 2024  Lecture 16 - ConvNets and TreeRNNs.srt":
        "CS224n_L16_I_ConvNets_and_TreeRNNs.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  Spring 2024  Lecture 18 - NLP, Linguistics, Philosophy.srt":
        "CS224n_L18_I_NLP_Linguistics_Philosophy.srt",

    # 2023 Lectures (additional content)
    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  2023  Lecture 8 - Self-Attention and Transformers.srt":
        "CS224n_L08_I_Self_Attention_and_Transformers.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  2023  Lecture 9 - Pretraining.srt":
        "CS224n_L09_I_Pretraining.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  2023  Lecture 11 - Natural Language Generation.srt":
        "CS224n_L17_I_Natural_Language_Generation.srt",  # Renumber to avoid conflict

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  2023  Lecture 16 - Multimodal Deep Learning, Douwe Kiela.srt":
        "CS224n_L19_I_Multimodal_Deep_Learning_by_Douwe_Kiela.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  2023  Lec. 19 - Model Interpretability & Editing, Been Kim.srt":
        "CS224n_L20_I_Model_Interpretability_and_Editing_by_Been_Kim.srt",

    # Tutorials
    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  2023  Python Tutorial, Manasi Sharma.srt":
        "CS224n_T01_I_Python_Tutorial_by_Manasi_Sharma.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP with Deep Learning  2023  PyTorch Tutorial,  Drew Kaul.srt":
        "CS224n_T02_I_PyTorch_Tutorial_by_Drew_Kaul.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS224N NLP w DL  2023  Hugging Face Tutorial, Eric Frankel.srt":
        "CS224n_T03_I_Hugging_Face_Tutorial_by_Eric_Frankel.srt"
}


def rename_cs224n_transcripts(transcript_dir):
    """Rename CS224n transcripts to systematic names"""

    transcript_path = Path(transcript_dir)

    if not transcript_path.exists():
        print(f"Error: Directory not found: {transcript_dir}")
        return

    print("="*60)
    print("RENAMING CS224N NLP TRANSCRIPTS")
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

    print(f"\n[SUCCESS] CS224n transcripts renamed and organized!")
    print(f"Location: {transcript_path}")


def create_metadata(transcript_path):
    """Create metadata file for CS224n transcripts"""

    metadata_content = """Stanford CS224n: Natural Language Processing with Deep Learning
========================================================================

Course Information:
- Course: Stanford CS224n NLP with Deep Learning
- Institution: Stanford University
- Instructors: Chris Manning, Shikhar Murty, and guest lecturers
- Playlist URL: https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D
- Total Lectures: 20 (Spring 2024: 16 lectures + 2023: 4 special lectures)
- Total Tutorials: 3 (Python, PyTorch, Hugging Face)
- Total Duration: ~30.6 hours

Spring 2024 Lectures (16):
--------------------------
01. Intro and Word Vectors
02. Word Vectors and Language Models
03. Backpropagation, Neural Networks
04. Dependency Parsing
05. Recurrent Neural Networks
06. Sequence to Sequence Models
07. Attention, Final Projects and LLM Intro
08. Self-Attention and Transformers (from 2023)
09. Pretraining (from 2023)
10. Post-training by Archit Sharma
11. Benchmarking by Yann Dubois
12. Efficient Training by Shikhar Murty
13. Brain-Computer Interfaces by Chaofei Fan
14. Reasoning and Agents by Shikhar Murty
15. After DPO by Nathan Lambert
16. ConvNets and TreeRNNs
17. Natural Language Generation (from 2023)
18. NLP, Linguistics, Philosophy
19. Multimodal Deep Learning by Douwe Kiela (from 2023)
20. Model Interpretability & Editing by Been Kim (from 2023)

Tutorials (3):
-----------
T01. Python Tutorial by Manasi Sharma
T02. PyTorch Tutorial by Drew Kaul
T03. Hugging Face Tutorial by Eric Frankel

Topics Covered:
--------------
- Word Vectors and Embeddings
- Neural Networks for NLP
- Backpropagation and Training
- Dependency Parsing
- Recurrent Neural Networks (RNNs)
- Sequence to Sequence Models
- Attention Mechanisms
- Transformers Architecture
- Self-Attention and Pretraining
- Post-training and Fine-tuning
- Natural Language Generation
- Benchmarking NLP Systems
- Efficient Training Methods
- Brain-Computer Interfaces
- Reasoning and AI Agents
- Reinforcement Learning (DPO)
- Convolutional Networks for NLP
- Tree-based Neural Networks
- Multimodal Deep Learning
- Model Interpretability
- Python, PyTorch, and Hugging Face Tools

File Naming Convention:
----------------------
CS224n_LXX_T_Topic.srt  (Lectures)
CS224n_TXY_T_Topic.srt  (Tutorials)

Where:
- XX: Lecture number (01-20)
- XY: Tutorial number (01-03)
- T: Type indicator (I = Lecture/Tutorial)
- Topic: Brief description of lecture content

Transcripts Source:
------------------
NoteGPT (30-second interval transcripts with timestamps)

Course Structure:
----------------
- Spring 2024: Latest lectures with current research
- 2023: Specialized topics and guest lectures
- Tutorials: Practical tools and frameworks

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

    metadata_file = transcript_path / "CS224n_Metadata.txt"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        print(f"\n[OK] Created metadata file: CS224n_Metadata.txt")
    except Exception as e:
        print(f"[ERROR] Failed to create metadata file: {e}")


if __name__ == "__main__":
    # Rename CS224n transcripts
    transcript_dir = "data/transcripts/cs224n"

    rename_cs224n_transcripts(transcript_dir)

    print("\nNext steps:")
    print("1. Run: python scripts/process_cs224n_pipeline.py")
    print("2. Run: python scripts/create_cs224n_index.py")
    print("3. Test: python scripts/test_cs224n_only.py")
