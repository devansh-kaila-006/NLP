"""
Rename MIT 6.S191 DL Main Transcripts to Systematic Names
"""

import os
from pathlib import Path

# File mappings: original NoteGPT name -> systematic name
FILE_MAPPINGS = {
    "NoteGPT_TRANSCRIPT_Lec 01. Introduction to Deep Learning.srt":
        "MIT_DL_L01_I_Introduction_to_Deep_Learning.srt",

    "NoteGPT_TRANSCRIPT_Lec 02. How to Train a Neural Net.srt":
        "MIT_DL_L02_I_How_to_Train_a_Neural_Net.srt",

    "NoteGPT_TRANSCRIPT_Lec 03. Approximation Theory.srt":
        "MIT_DL_L03_I_Approximation_Theory.srt",

    "NoteGPT_TRANSCRIPT_Lec 04. Architectures Grids.srt":
        "MIT_DL_L04_I_Architectures_Grids.srt",

    "NoteGPT_TRANSCRIPT_Lec 05. Architectures Graphs.srt":
        "MIT_DL_L05_I_Architectures_Graphs.srt",

    "NoteGPT_TRANSCRIPT_Lec 06. Generalization Theory.srt":
        "MIT_DL_L06_I_Generalization_Theory.srt",

    "NoteGPT_TRANSCRIPT_Lec 07. Scaling Rules for Optimization.srt":
        "MIT_DL_L07_I_Scaling_Rules_for_Optimization.srt",

    "NoteGPT_TRANSCRIPT_Lec 08. Architectures Transformers.srt":
        "MIT_DL_L08_I_Architectures_Transformers.srt",

    "NoteGPT_TRANSCRIPT_Lec 09. Hacker's Guide to Deep Learning.srt":
        "MIT_DL_L09_I_Hackers_Guide_to_Deep_Learning.srt",

    "NoteGPT_TRANSCRIPT_Lec 10. Architectures Memory.srt":
        "MIT_DL_L10_I_Architectures_Memory.srt",

    "NoteGPT_TRANSCRIPT_Lec 11. Representation Learning Reconstruction-Based.srt":
        "MIT_DL_L11_I_Representation_Learning_Reconstruction_Based.srt",

    "NoteGPT_TRANSCRIPT_Lec 12. Representation Learning Similarity-Based.srt":
        "MIT_DL_L12_I_Representation_Learning_Similarity_Based.srt",

    "NoteGPT_TRANSCRIPT_Lec 13. Representation Learning Theory.srt":
        "MIT_DL_L13_I_Representation_Learning_Theory.srt",

    "NoteGPT_TRANSCRIPT_Lec 14. Generative Models Basics.srt":
        "MIT_DL_L14_I_Generative_Models_Basics.srt",

    "NoteGPT_TRANSCRIPT_Lec 15. Generative Models Representation Learning Meets Generative Modeling.srt":
        "MIT_DL_L15_I_Generative_Models_Rep_Learning_Meets_Gen_Modeling.srt",

    "NoteGPT_TRANSCRIPT_Lec 16. Generative Models Conditional Models.srt":
        "MIT_DL_L16_I_Generative_Models_Conditional_Models.srt",

    "NoteGPT_TRANSCRIPT_Lec 17. Generalization Out-of-Distribution (OOD).srt":
        "MIT_DL_L17_I_Generalization_Out_of_Distribution_OOD.srt",

    "NoteGPT_TRANSCRIPT_Lec 18. Transfer Learning Models.srt":
        "MIT_DL_L18_I_Transfer_Learning_Models.srt",

    "NoteGPT_TRANSCRIPT_Lec 19. Transfer Learning Data.srt":
        "MIT_DL_L19_I_Transfer_Learning_Data.srt",

    "NoteGPT_TRANSCRIPT_Lec 20. Scaling Laws.srt":
        "MIT_DL_L20_I_Scaling_Laws.srt",

    "NoteGPT_TRANSCRIPT_Lec 21. Language Models.srt":
        "MIT_DL_L21_I_Language_Models.srt",

    "NoteGPT_TRANSCRIPT_Lec 23. Metrized Deep Learning.srt":
        "MIT_DL_L23_I_Metrized_Deep_Learning.srt",

    "NoteGPT_TRANSCRIPT_Lec 24. Inference Methods for Deep Learning.srt":
        "MIT_DL_L24_I_Inference_Methods_for_Deep_Learning.srt",

    "NoteGPT_TRANSCRIPT_PyTorch Tutorial.srt":
        "MIT_DL_T01_I_PyTorch_Tutorial.srt"
}


def rename_mit_dl_main_transcripts(transcript_dir):
    """Rename MIT DL main transcripts to systematic names"""

    transcript_path = Path(transcript_dir)

    if not transcript_path.exists():
        print(f"Error: Directory not found: {transcript_dir}")
        return

    print("="*60)
    print("RENAMING MIT 6.S191 DL MAIN TRANSCRIPTS")
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

    print(f"\n[SUCCESS] MIT DL main transcripts renamed and organized!")
    print(f"Location: {transcript_path}")


def create_metadata(transcript_path):
    """Create metadata file for MIT DL main transcripts"""

    metadata_content = """MIT 6.S191: Introduction to Deep Learning (Main Playlist)
========================================================================

Course Information:
- Course: MIT 6.S191 Introduction to Deep Learning
- Institution: Massachusetts Institute of Technology
- Instructor: MIT
- Playlist URL: https://www.youtube.com/playlist?list=PLUl4u3cNGP63URZnh5iqBzDTDYPUTQT-8
- Total Lectures: 24
- Total Tutorials: 1 (PyTorch)
- Total Duration: ~30+ hours

Lecture List:
-------------
01. Introduction to Deep Learning
02. How to Train a Neural Net
03. Approximation Theory
04. Architectures Grids
05. Architectures Graphs
06. Generalization Theory
07. Scaling Rules for Optimization
08. Architectures Transformers
09. Hacker's Guide to Deep Learning
10. Architectures Memory
11. Representation Learning Reconstruction-Based
12. Representation Learning Similarity-Based
13. Representation Learning Theory
14. Generative Models Basics
15. Generative Models Representation Learning Meets Generative Modeling
16. Generative Models Conditional Models
17. Generalization Out-of-Distribution (OOD)
18. Transfer Learning Models
19. Transfer Learning Data
20. Scaling Laws
21. Language Models
22. [Missing from playlist]
23. Metrized Deep Learning
24. Inference Methods for Deep Learning

Tutorials (1):
-----------
T01. PyTorch Tutorial

Topics Covered:
--------------
- Deep Learning Fundamentals
- Neural Network Training
- Approximation Theory
- Neural Network Architectures (Grids, Graphs, Transformers, Memory)
- Generalization Theory and OOD Detection
- Optimization Scaling Rules
- Representation Learning (Reconstruction-Based, Similarity-Based)
- Generative Models (Basics, Conditional, Representation Learning)
- Transfer Learning (Models, Data)
- Scaling Laws
- Language Models
- Metrized Deep Learning
- Inference Methods
- PyTorch Framework

File Naming Convention:
----------------------
MIT_DL_LXX_T_Topic.srt  (Lectures)
MIT_DL_TXY_T_Topic.srt  (Tutorials)

Where:
- XX: Lecture number (01-24)
- XY: Tutorial number (01-01)
- T: Type indicator (I = Lecture/Tutorial)
- Topic: Brief description of lecture content

Transcripts Source:
------------------
NoteGPT (30-second interval transcripts with timestamps)

Course Structure:
----------------
This is the MAIN MIT 6.S191 playlist with comprehensive coverage:
- 24 lectures (vs 11 in dl_alt)
- 1 tutorial
- More advanced topics
- Comprehensive deep learning curriculum

Processing Status:
----------------
[ ] Upload transcripts (completed)
[ ] Rename transcripts (completed)
[ ] Process video chunks
[ ] Generate embeddings
[ ] Create FAISS index
[ ] Test multi-modal queries

Last Updated: 2026-05-17
"""

    metadata_file = transcript_path / "MIT_DL_Main_Metadata.txt"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        print(f"\n[OK] Created metadata file: MIT_DL_Main_Metadata.txt")
    except Exception as e:
        print(f"[ERROR] Failed to create metadata file: {e}")


if __name__ == "__main__":
    # Rename MIT DL main transcripts
    transcript_dir = "data/transcripts/dl"

    rename_mit_dl_main_transcripts(transcript_dir)

    print("\nNext steps:")
    print("1. Run: python scripts/process_mit_dl_main_pipeline.py")
    print("2. Run: python scripts/create_mit_dl_main_index.py")
    print("3. Test: python scripts/test_final_multimodal.py")
    print("\nThis adds the COMPREHENSIVE MIT DL playlist (24 lectures)!")
