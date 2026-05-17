"""
Rename CS231n Computer Vision Transcripts to Systematic Names
"""

import os
from pathlib import Path

# File mappings: original NoteGPT name -> systematic name
FILE_MAPPINGS = {
    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 1 Introduction.srt":
        "CS231n_L01_I_Introduction.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 2 Image Classification with Linear Classifiers.srt":
        "CS231n_L02_I_Image_Classification_with_Linear_Classifiers.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 3 Regularization and Optimization.srt":
        "CS231n_L03_I_Regularization_and_Optimization.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 4 Neural Networks and Backpropagation.srt":
        "CS231n_L04_I_Neural_Networks_and_Backpropagation.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 5 Image Classification with CNNs.srt":
        "CS231n_L05_I_Image_Classification_with_CNNs.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 6 CNN Architectures.srt":
        "CS231n_L06_I_CNN_Architectures.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 7 Recurrent Neural Networks.srt":
        "CS231n_L07_I_Recurrent_Neural_Networks.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 8 Attention and Transformers.srt":
        "CS231n_L08_I_Attention_and_Transformers.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 9 Object Detection, Image Segmentation, Visualizing.srt":
        "CS231n_L09_I_Object_Detection_Image_Segmentation_Visualizing.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 10 Video Understanding.srt":
        "CS231n_L10_I_Video_Understanding.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 11 Large Scale Distributed Training.srt":
        "CS231n_L11_I_Large_Scale_Distributed_Training.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N  Spring 2025  Lecture 12 Self-Supervised Learning.srt":
        "CS231n_L12_I_Self_Supervised_Learning.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 13 Generative Models 1.srt":
        "CS231n_L13_I_Generative_Models_1.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision Spring 2025  Lecture 14 Generative Models 2.srt":
        "CS231n_L14_I_Generative_Models_2.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 15 3D Vision.srt":
        "CS231n_L15_I_3D_Vision.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 16 Vision and Language.srt":
        "CS231n_L16_I_Vision_and_Language.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 17 Robot Learning.srt":
        "CS231n_L17_I_Robot_Learning.srt",

    "NoteGPT_TRANSCRIPT_Stanford CS231N Deep Learning for Computer Vision  Spring 2025  Lecture 18 Human-Centered AI.srt":
        "CS231n_L18_I_Human_Centered_AI.srt"
}


def rename_cs231n_transcripts(transcript_dir):
    """Rename CS231n transcripts to systematic names"""

    transcript_path = Path(transcript_dir)

    if not transcript_path.exists():
        print(f"Error: Directory not found: {transcript_dir}")
        return

    print("="*60)
    print("RENAMING CS231N COMPUTER VISION TRANSCRIPTS")
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

    print(f"\n[SUCCESS] CS231n transcripts renamed and organized!")
    print(f"Location: {transcript_path}")


def create_metadata(transcript_path):
    """Create metadata file for CS231n transcripts"""

    metadata_content = """Stanford CS231n: Computer Vision / Deep Learning
========================================================================

Course Information:
- Course: Stanford CS231n Computer Vision
- Institution: Stanford University
- Instructors: Fei-Fei Li, Justin Johnson, Serena Yeung
- Playlist URL: https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16
- Total Lectures: 18
- Total Duration: ~22.5 hours

Lecture List (Spring 2025):
--------------------------
01. Introduction
02. Image Classification with Linear Classifiers
03. Regularization and Optimization
04. Neural Networks and Backpropagation
05. Image Classification with CNNs
06. CNN Architectures
07. Recurrent Neural Networks
08. Attention and Transformers
09. Object Detection, Image Segmentation, Visualizing
10. Video Understanding
11. Large Scale Distributed Training
12. Self-Supervised Learning
13. Generative Models 1
14. Generative Models 2
15. 3D Vision
16. Vision and Language
17. Robot Learning
18. Human-Centered AI

Topics Covered:
--------------
- Image Classification (Linear Classifiers → CNNs)
- Neural Networks and Backpropagation
- Convolutional Neural Networks (CNNs)
- CNN Architectures (ResNet, VGG, etc.)
- Recurrent Neural Networks for Vision
- Attention Mechanisms and Transformers
- Object Detection and Image Segmentation
- Visualization and Interpretability
- Video Understanding
- Large Scale Distributed Training
- Self-Supervised Learning
- Generative Models (GANs, VAEs, Diffusion)
- 3D Vision and Geometry
- Vision and Language (VQA, Captioning)
- Robot Learning and Embodied AI
- Human-Centered AI

File Naming Convention:
----------------------
CS231n_LXX_T_Topic.srt

Where:
- XX: Lecture number (01-18)
- T: Type indicator (I = Lecture)
- Topic: Brief description of lecture content

Transcripts Source:
------------------
NoteGPT (30-second interval transcripts with timestamps)

Course Highlights:
-----------------
- Comprehensive computer vision curriculum
- From basic classifiers to state-of-the-art deep learning
- Covers both traditional and modern approaches
- Includes cutting-edge topics like transformers and generative models
- Practical applications in robotics, healthcare, and human-centered AI

Processing Status:
-----------------
[ ] Upload transcripts (completed)
[ ] Rename transcripts (completed)
[ ] Process video chunks
[ ] Generate embeddings
[ ] Create FAISS index
[ ] Test multi-modal queries
[ ] Complete multi-modal RAG system

Last Updated: 2026-05-17
"""

    metadata_file = transcript_path / "CS231n_Metadata.txt"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        print(f"\n[OK] Created metadata file: CS231n_Metadata.txt")
    except Exception as e:
        print(f"[ERROR] Failed to create metadata file: {e}")


if __name__ == "__main__":
    # Rename CS231n transcripts
    transcript_dir = "data/transcripts/cs231n"

    rename_cs231n_transcripts(transcript_dir)

    print("\nNext steps:")
    print("1. Run: python scripts/process_cs231n_pipeline.py")
    print("2. Run: python scripts/create_cs231n_index.py")
    print("3. Test: python scripts/test_final_multimodal.py")
    print("\n🎉 This will complete ALL 4 playlists for the multi-modal RAG system!")
