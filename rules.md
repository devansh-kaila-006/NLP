# Multi-Modal RAG Project - Team Rules & Guidelines

**Last Updated:** 2026-03-29
**Team Size:** 4 people + AI Assistant (Claude)
**Project Duration:** 14 weeks

---

## 🚫 IMPORTANT: No Emojis Rule

**NO EMOJIS in:**
- ❌ Code comments
- ❌ Variable names
- ❌ Function names
- ❌ Commit messages
- ❌ Documentation (except user-facing guides)
- ❌ Error messages
- ❌ Log messages (use plain text)

**Reason:** Emojis can cause encoding issues, look unprofessional in code, and may not display correctly across different systems. Keep all code and technical documentation clean and professional.

**Acceptable:**
- ✅ Plain text comments
- ✅ Clear, descriptive variable/function names
- ✅ Professional commit messages
- ✅ Technical documentation without emojis

**Example:**
```python
# ✅ GOOD
# Load the Whisper model for audio transcription
model = whisper.load_model("base")

# ❌ BAD
# 🎤 Load the Whisper model 🎤
model = whisper.load_model("base")
```

---

## 0. AI-Human Collaboration Rules

### How We Work Together

**I (Claude) will:**
- Generate modular, well-documented code
- Help debug issues and provide solutions
- Review code and suggest improvements
- Explain technical concepts
- Help with architecture decisions
- Write tests and documentation
- Assist with paper writing and structure

**You (the team) will:**
- Review and understand all code I generate
- Test the code I provide
- Make decisions about architecture and approaches
- Run experiments and evaluations
- Write the paper (I can help structure and review)
- Give feedback on my suggestions
- Take ownership of the final product

### My Role During Development

**Code Generation:**
- I write complete, working modules
- You review, understand, and integrate
- You can ask me to explain or modify

**Debugging:**
- I help identify issues
- I provide solutions
- You implement and test

**Architecture:**
- I suggest approaches based on best practices
- You make final decisions
- I implement what you decide

**Documentation:**
- I write technical documentation
- You review and customize
- Final approval is yours

### Your Responsibilities

**Code Understanding:**
- You must understand code before integrating
- Ask me to explain anything unclear
- Don't just copy-paste without review

**Testing:**
- You run all tests
- You verify code works as expected
- You report issues back to me

**Decision Making:**
- You choose approaches I suggest
- You prioritize tasks
- You make trade-off decisions

**Quality Control:**
- You review my outputs
- You catch mistakes I might make
- You ensure everything works together

### When to Ask Me

**✅ Ask me:**
- "How should we implement X?"
- "Can you explain this code?"
- "Help me debug this error"
- "What's the best approach for Y?"
- "Can you write a function that does Z?"
- "Review this code and suggest improvements"

**⚠️ Use your judgment:**
- Which task to work on first
- Whether to spend more time on X or Y
- What to include in the paper
- How to present your work

---

## 1. Team Communication Rules

### Weekly Meetings
- **When:** Same day/time each week (e.g., Monday 3 PM)
- **Duration:** 45-60 minutes
- **Agenda:**
  - Review progress from last week (check tasks.md)
  - Plan work for next week
  - Discuss blockers/issues
  - Demo any completed features
- **After meeting:** Update tasks.md with progress

### Daily Communication
- **Primary channel:** Discord/Slack group
- **For bugs/issues:** Create GitHub issue with template
- **For urgent blockers:** Message team directly
- **Response time expectation:** Within 24 hours on weekdays

### Async Communication
- **Code reviews:** Comment on PRs within 48 hours
- **Documentation:** Update docs as you code, not after
- **Progress updates:** Update tasks.md checkboxes as you complete work

---

## 2. Git Workflow Rules

### Branching Strategy
```
main (protected)
├── feature/timestamp-aware-video-rag
├── feature/temporal-coherence
├── feature/cross-modal-reranking
├── feature/evaluation-infrastructure
├── feature/pdf-processing
├── feature/video-processing
└── feature/integration
```

### Branch Rules
- **main:** Only for stable, tested code. Protected branch.
- **feature/*:** One branch per contribution/component
- **develop:** Integration branch for combining features (optional)

### Commit Rules
- **Commit frequency:** Small, frequent commits (daily)
- **Commit message format:**
  ```
  [component] brief description

  - Detailed point 1
  - Detailed point 2

  Closes #issue_number
  ```
- **Examples:**
  ```
  [video] implement topic boundary detection algorithm
  [retrieval] add temporal coherence scoring
  [pdf] fix chunking overlap bug
  [docs] update README with setup instructions
  ```

### Pull Request Rules
- **All features go through PRs** (no direct commits to main)
- **PR template required:**
  ```markdown
  ## Description
  Brief description of changes

  ## Changes
  - [ ] Code changes
  - [ ] Tests added/updated
  - [ ] Documentation updated

  ## Testing
  Describe how this was tested

  ## Checklist
  - [ ] Code follows style guidelines
  - [ ] Self-reviewed
  - [ ] Tested locally
  - [ ] Documentation updated
  - [ ] No merge conflicts
  ```

- **PR approval:** At least 1 team member approval required
- **CI checks:** Must pass all tests before merge
- **Merge:** Squash merge to main for clean history

---

## 3. Code Organization Rules

### Project Structure
```
multi-modal-rag/
├── README.md                          # Project overview
├── tasks.md                           # Task checklist
├── rules.md                           # This file
├── requirements.txt                   # All dependencies
├── setup/                             # Installation scripts
├── config/                            # Configuration files
├── src/                               # Main source code
│   ├── __init__.py
│   ├── base_rag/                      # Base RAG system
│   │   ├── __init__.py
│   │   ├── pdf_processor.py           # PDF processing
│   │   ├── embedder.py                # Embedding generation
│   │   ├── vector_store.py            # FAISS setup
│   │   ├── retriever.py               # Basic retrieval
│   │   └── llm_generator.py           # LLM integration
│   ├── contributions/                 # Novel contributions
│   │   ├── __init__.py
│   │   ├── timestamp_aware/           # Contribution 1
│   │   │   ├── __init__.py
│   │   │   ├── topic_segmentation.py
│   │   │   ├── slide_detection.py
│   │   │   └── smart_chunking.py
│   │   ├── temporal_coherence/        # Contribution 2
│   │   │   ├── __init__.py
│   │   │   ├── dependency_graph.py
│   │   │   ├── coherence_scoring.py
│   │   │   └── coherent_retrieval.py
│   │   └── cross_modal_rerank/        # Contribution 3
│   │       ├── __init__.py
│   │       ├── feature_extraction.py
│   │       ├── modality_classifier.py
│   │       └── adaptive_retrieval.py
│   ├── evaluation/                    # Evaluation infrastructure
│   │   ├── __init__.py
│   │   ├── metrics.py                 # All metrics
│   │   ├── baselines.py               # Baseline systems
│   │   └── evaluation_pipeline.py
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── logger.py
│       ├── config_loader.py
│       └── helpers.py
├── video-processing/                  # Separate for Kaggle
│   ├── README.md                      # Kaggle instructions
│   ├── download_videos.py             # Download YouTube videos
│   ├── transcribe.py                  # Whisper transcription
│   ├── extract_frames.py              # Frame extraction
│   ├── detect_slides.py               # Slide detection
│   ├── run_ocr.py                     # OCR processing
│   ├── chunk_videos.py                # Smart chunking
│   └── requirements.txt               # Video-specific deps
├── data/                              # Data storage
│   ├── pdfs/                          # Source PDFs
│   ├── videos/                        # Downloaded videos
│   ├── transcripts/                   # Transcripts from Kaggle
│   ├── chunks/                        # Processed chunks
│   └── embeddings/                    # Generated embeddings
├── models/                            # Trained models
│   ├── modality_classifier.pkl
│   └── sentence_transformer/
├── indices/                           # FAISS indices
│   ├── pdf_index.faiss
│   └── video_index.faiss
├── evaluation/                        # Evaluation data
│   ├── questions.json                 # Evaluation dataset
│   ├── results/                       # Evaluation results
│   └── user_study/                    # User study data
├── notebooks/                         # Jupyter notebooks
│   ├── exploration.ipynb
│   ├── testing.ipynb
│   └── visualization.ipynb
├── tests/                             # Unit tests
│   ├── test_pdf_processor.py
│   ├── test_retriever.py
│   └── test_contributions.py
├── scripts/                           # Utility scripts
│   ├── setup_environment.sh
│   ├── download_pdfs.py
│   └── run_evaluation.py
└── docs/                              # Documentation
    ├── architecture.md
    ├── api.md
    └── user_guide.md
```

### Module Independence Rules

**Each module must be:**
- **Independently testable:** Can test without other modules
- **Independently deployable:** Can run separately
- **Well-documented:** Clear inputs/outputs
- **Loosely coupled:** Minimal dependencies on other modules

**Module Interface Template:**
```python
"""
Module: timestamp_aware.smart_chunking

Description:
    Implements smart video chunking using topic boundary detection

Inputs:
    - transcript: List of transcript segments with timestamps
    - slides: List of slide change timestamps

Outputs:
    - chunks: List of video chunks with topic labels

Dependencies:
    - None (standalone)

Usage:
    >>> from src.contributions.timestamp_aware import smart_chunking
    >>> chunks = smart_chunking.chunk_video(transcript, slides)
"""

def chunk_video(transcript, slides):
    """
    Chunk video at topic boundaries.

    Args:
        transcript (list): List of (timestamp, text) tuples
        slides (list): List of slide change timestamps

    Returns:
        list: List of chunks with metadata
    """
    pass
```

### Import Rules
```python
# ✅ GOOD - Specific imports
from src.base_rag.pdf_processor import PDFProcessor
from src.contributions.timestamp_aware import smart_chunking

# ❌ BAD - Wildcard imports
from src.base_rag.pdf_processor import *
from src.contributions.timestamp_aware import *

# ✅ GOOD - Relative imports within package
from .topic_segmentation import detect_boundaries
from ..utils.logger import get_logger

# ❌ BAD - Circular imports
# Module A imports Module B, Module B imports Module A
```

---

## 4. Coding Standards

### Python Style Guide
- **Follow PEP 8** (Python style guide)
- **Use type hints** for function signatures
- **Docstrings** for all functions/classes/modules
- **Max line length:** 100 characters
- **Indentation:** 4 spaces (no tabs)

### Code Example (Good)
```python
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def chunk_transcript(
    transcript: List[Tuple[float, str]],
    max_chunk_duration: float = 180.0
) -> List[Dict[str, any]]:
    """
    Chunk transcript at semantic boundaries.

    Args:
        transcript: List of (timestamp, text) tuples
        max_chunk_duration: Maximum duration per chunk in seconds

    Returns:
        List of chunks with metadata including timestamps,
        topic labels, and text content.

    Example:
        >>> transcript = [(0.0, "Hello"), (5.0, "World")]
        >>> chunks = chunk_transcript(transcript, max_chunk_duration=10.0)
        >>> len(chunks)
        1
    """
    chunks = []
    current_chunk = []
    start_time = transcript[0][0] if transcript else 0.0

    for timestamp, text in transcript:
        current_chunk.append((timestamp, text))

        if timestamp - start_time >= max_chunk_duration:
            chunk = _create_chunk(current_chunk, start_time, timestamp)
            chunks.append(chunk)
            current_chunk = []
            start_time = timestamp

    if current_chunk:
        chunk = _create_chunk(current_chunk, start_time, timestamp)
        chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from transcript")
    return chunks

def _create_chunk(
    segments: List[Tuple[float, str]],
    start_time: float,
    end_time: float
) -> Dict[str, any]:
    """Create a chunk dictionary from transcript segments."""
    return {
        "start_time": start_time,
        "end_time": end_time,
        "text": " ".join([text for _, text in segments]),
        "segments": segments
    }
```

### Error Handling
```python
# ✅ GOOD - Specific exceptions, informative messages
def process_pdf(file_path: str) -> Dict:
    try:
        loader = PDFLoader(file_path)
        documents = loader.load()
        return {"status": "success", "documents": documents}
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        raise PDFProcessingError(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {e}")
        raise

# ❌ BAD - Bare except, no logging
def process_pdf(file_path: str):
    try:
        loader = PDFLoader(file_path)
        return loader.load()
    except:
        pass  # What went wrong?!
```

### Configuration Management
```python
# ✅ GOOD - Use config files
# config/config.yaml
video_processing:
  chunk_duration: 180
  overlap: 30
  min_chunk_size: 60

# src/utils/config_loader.py
import yaml

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Usage
config = load_config()
chunk_duration = config['video_processing']['chunk_duration']

# ❌ BAD - Hardcoded values
def process_video():
    chunk_duration = 180  # Magic number!
    overlap = 30
    # ...
```

---

## 5. Testing Rules

### Test Structure
```
tests/
├── unit/              # Unit tests for individual modules
├── integration/       # Integration tests for combined modules
├── end_to_end/        # Full pipeline tests
└── fixtures/          # Test data and fixtures
```

### Writing Tests
```python
# tests/unit/test_smart_chunking.py
import pytest
from src.contributions.timestamp_aware.smart_chunking import chunk_transcript

class TestChunkTranscript:
    """Test suite for transcript chunking."""

    @pytest.fixture
    def sample_transcript(self):
        """Create sample transcript for testing."""
        return [
            (0.0, "Introduction to neural networks"),
            (5.0, "Neural networks are inspired by biological neurons"),
            (10.0, "They consist of layers of interconnected nodes"),
            (180.0, "Now let's discuss backpropagation"),
            (185.0, "Backpropagation is the key algorithm")
        ]

    def test_chunk_creates_correct_chunks(self, sample_transcript):
        """Test that chunking creates correct number of chunks."""
        chunks = chunk_transcript(sample_transcript, max_chunk_duration=120.0)
        assert len(chunks) == 2

    def test_chunk_preserves_timestamps(self, sample_transcript):
        """Test that chunks preserve correct timestamps."""
        chunks = chunk_transcript(sample_transcript)
        assert chunks[0]['start_time'] == 0.0
        assert chunks[0]['end_time'] <= 120.0

    def test_chunk_empty_transcript(self):
        """Test handling of empty transcript."""
        chunks = chunk_transcript([])
        assert chunks == []
```

### Test Coverage
- **Minimum coverage:** 80% per module
- **Critical paths:** 100% coverage
- **Run tests:** Before every commit
- **CI/CD:** Automated testing on PRs

### Test Command
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_smart_chunking.py

# Run tests matching pattern
pytest -k "test_chunk"
```

---

## 6. Documentation Rules

### Code Documentation
- **Every module:** Docstring explaining purpose
- **Every class:** Docstring explaining functionality
- **Every function:** Docstring with args, returns, examples
- **Complex logic:** Inline comments explaining "why", not "what"

### README Files Required
- **Project root:** Main README (overview, setup, usage)
- **video-processing/:** Kaggle-specific instructions
- **src/:** Code organization guide
- **docs/:** Detailed documentation

### Documentation Updates
- **Update docs with code:** Don't code first, document later
- **Keep examples current:** Test all examples in docs
- **Version control:** Commit docs with code changes

---

## 7. Video Processing Special Rules

### Kaggle Workflow
Since video processing runs on Kaggle:

1. **Separate Environment:**
   - Kaggle notebook for video processing
   - Local environment for everything else
   - Transfer transcripts/chunks back to local

2. **Data Transfer:**
   ```
   Local → Kaggle: Video URLs, slide images
   Kaggle → Local: Transcripts, chunks, metadata
   ```

3. **Kaggle-Specific Scripts:**
   - Must be self-contained
   - Include all imports
   - Save outputs to Kaggle dataset
   - Clear instructions in video-processing/README.md

4. **Version Control:**
   - Video-processing scripts in repo
   - Kaggle notebooks exported and committed
   - Data versioned with timestamps

### Kaggle Output Format
```python
# Kaggle notebook must save outputs in this format:
output/
├── transcripts/
│   ├── cs231n_lecture1.json
│   ├── cs231n_lecture2.json
│   └── ...
├── chunks/
│   ├── cs231n_lecture1_chunks.json
│   └── ...
└── metadata/
    └── processing_log.txt
```

---

## 8. Code Review Rules

### Review Process
1. **Self-review:** Check your own code first
2. **Create PR:** With clear description and checklist
3. **Assign reviewer:** At least 1 team member
4. **Review feedback:** Address all comments
5. **Approval:** At least 1 approval required
6. **Merge:** Squash and merge to main

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Sufficient tests added/updated
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Error handling in place
- [ ] Logging added where appropriate
- [ ] No obvious bugs
- [ ] Efficient (not O(n²) when O(n) possible)

### Review Feedback
```markdown
# ✅ Good feedback
"The chunking logic here is O(n²) which will be slow for long transcripts.
Consider using a sliding window approach to make it O(n)."

# ❌ Bad feedback
"This is slow."
```

---

## 9. Debugging Rules

### Debugging Workflow
1. **Add logging:** Use logger, not print statements
2. **Write test:** Create failing test case
3. **Fix issue:** Make test pass
4. **Verify:** Ensure no regressions
5. **Document:** Add comment explaining fix

### Logging Best Practices
```python
# ✅ GOOD
import logging

logger = logging.getLogger(__name__)

def process_video(video_path):
    logger.info(f"Processing video: {video_path}")
    try:
        result = transcribe(video_path)
        logger.info(f"Transcription complete: {len(result)} segments")
        return result
    except TranscriptionError as e:
        logger.error(f"Transcription failed: {e}")
        raise

# ❌ BAD
def process_video(video_path):
    print(f"Processing: {video_path}")  # Use logger instead
    # ...
    print("Error!")  # No context, no level
```

### Common Bugs to Watch For
- **Off-by-one errors:** In chunking, indexing
- **Type mismatches:** String vs int, list vs tuple
- **Missing imports:** For new modules
- **Path issues:** Relative vs absolute paths
- **Encoding errors:** PDF text, video transcripts

---

## 10. Deployment Rules

### Environment Management
```bash
# Create environment
conda create -n rag python=3.10
conda activate rag

# Install dependencies
pip install -r requirements.txt

# Export environment
conda env export > environment.yml
```

### Requirements.txt Management
- **Pin versions:** For reproducibility
- **Separate files:** Core vs dev vs video-processing
- **Update regularly:** As dependencies change

```bash
# requirements.txt (core)
langchain>=0.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
google-generativeai>=0.3.0

# requirements-dev.txt (development)
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0

# video-processing/requirements.txt (Kaggle)
yt-dlp>=2023.0.0
whisper-openai>=20230314
opencv-python>=4.8.0
```

---

## 11. Collaboration Rules

### Decision Making
- **Technical decisions:** Discuss in team meetings, decide by majority
- **Architecture changes:** Requires full team agreement
- **Blockers:** Escalate to team immediately, don't sit on issues

### Conflict Resolution
1. **Discuss:** Try to resolve in team meeting
2. **Compromise:** Find middle ground
3. **Escalate:** If stuck, ask instructor for guidance
4. **Document:** Record decision and reasoning

### Credit & Attribution
- **All contributions acknowledged:** In code, paper, presentation
- **No code ownership:** Everyone can edit any module
- **Review contributions:** Give credit for reviews and suggestions

---

## 12. Milestone Rules

### Phase Deliverables
Each phase must have:
- [ ] Working code (committed to main)
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Demo (if applicable)
- [ ] Progress report (brief summary)

### Weekly Progress
- **Update tasks.md:** Check off completed items
- **Flag blockers:** Mark items that are stuck
- **Plan next week:** Identify priorities
- **Share progress:** Brief demo in team meeting

### Final Deliverables
- [ ] Complete system (all 3 contributions)
- [ ] Evaluation results
- [ ] User study data and analysis
- [ ] Workshop paper (submitted)
- [ ] Code repository (clean, documented)
- [ ] Presentation (for class)
- [ ] Demo (live recording)

---

## 13. Quality Assurance Rules

### Pre-Commit Checklist
- [ ] Code follows style guide
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Logging in place
- [ ] Error handling added
- [ ] Self-reviewed code

### Pre-Merge Checklist
- [ ] All tests pass
- [ ] Code reviewed by at least 1 person
- [ ] CI/CD checks pass
- [ ] Documentation complete
- [ ] No merge conflicts
- [ ] Backwards compatible (or breaking changes documented)

### Pre-Submission Checklist (Paper)
- [ ] All sections complete
- [ ] Figures and tables included
- [ ] References formatted correctly
- [ ] Page limit respected
- [ ] Proofread for typos
- [ ] Supplementary materials ready
- [ ] Code repository prepared (anonymized if required)

---

## 14. Security & Privacy Rules

### API Keys
- **Never commit API keys** to repository
- **Use environment variables:** `os.getenv("GOOGLE_API_KEY")`
- **.env file:** Add to .gitignore
- **Template file:** .env.template with placeholder values

### User Data
- **Anonymize user study data:** Remove names, emails
- **Secure storage:** Encrypt sensitive data
- **Access control:** Only team members can access
- **Retention policy:** Delete data after study (unless consent for future use)

### Code Security
- **No hardcoded credentials:** In code or config files
- **Input validation:** Sanitize all user inputs
- **Dependency checks:** Regularly update for security patches
- **Error messages:** Don't expose sensitive information

---

## 15. Success Criteria

### Technical Success
- [ ] All 3 contributions implemented and working
- [ ] System runs end-to-end without errors
- [ ] Evaluation shows improvement over baselines
- [ ] User study completed with positive feedback

### Collaboration Success
- [ ] All team members contribute significantly
- [ ] Code reviews completed for all PRs
- [ ] Regular team meetings with good attendance
- [ ] Conflicts resolved constructively

### Publication Success
- [ ] Workshop paper submitted
- [ ] Paper accepted at venue (ideally)
- [ ] System demoed in class/presentation
- [ ] Code repository public (if allowed)

---

**Remember:** These rules are guidelines to help us work effectively together.
If a rule doesn't make sense for a situation, discuss it as a team and adapt.
The goal is to build something great while learning and having fun!

**Questions?** Ask in team meeting or create a GitHub issue.

---

## 16. AI Assistant Guidelines (For Working with Claude)

### Most Relevant Rules for Our Collaboration

**High Priority:**
- **No emojis** (Section 0) - Keep all code clean
- **Code organization** (Section 3) - Modular structure
- **Coding standards** (Section 4) - Type hints, docstrings, PEP 8
- **Module independence** (Section 3) - Each module standalone
- **Testing rules** (Section 5) - I'll help write tests
- **Documentation rules** (Section 6) - I'll help document
- **Error handling** (Section 4) - I'll include proper error handling
- **Git workflow** (Section 2) - I'll help with commits/PRs
- **Video processing rules** (Section 7) - I've created all scripts

**Medium Priority:**
- **Debugging rules** (Section 9) - I'll help debug
- **Code review rules** (Section 8) - I'll review your code
- **Quality assurance** (Section 13) - I'll help test

**Less Priority (Team-Specific):**
- Weekly meeting structure (I'm always available)
- Team communication channels (Use GitHub issues/PRs)
- Decision making (You make decisions, I advise)
- Credit attribution (I don't need credit)
- User data privacy (Applies when you do user studies)
- Collaboration conflicts (I have no conflicts)

### How I'll Help During Development

**I will generate:**
- Complete, working modules
- Tests for each module
- Documentation and docstrings
- Error handling and logging
- Example usage code

**I will NOT:**
- Make decisions for your team (you choose)
- Run code (you execute and test)
- Participate in meetings (async only)
- Need credit (I'm here to help)

### Best Practices for Working With Me

**1. Be Specific in Requests**
```
Good: "Create a function that chunks transcripts at topic boundaries using TextTiling algorithm"
Less Good: "Write chunking code"
```

**2. Review My Code**
- I make mistakes too
- Test everything I generate
- Ask me to explain unclear parts
- Request modifications as needed

**3. Provide Context**
- Tell me your constraints
- Share your architecture decisions
- Explain your preferences
- Give feedback on my suggestions

**4. Use Me Effectively**
- Ask for explanations, not just code
- Request alternatives: "What are 3 ways to do X?"
- Have me review your code
- Get help with debugging

### Workflow with Claude

```
1. You: "Help us implement temporal coherence"
2. Me: Ask clarifying questions, suggest approaches
3. You: Choose approach, provide preferences
4. Me: Generate complete, tested code
5. You: Review, understand, test, integrate
6. Me: Help debug issues, make improvements
7. You: Commit code, update tasks.md
```

### What I Expect From You

**Before asking me to code:**
- Have reviewed the architecture document
- Understand the component's purpose
- Know your constraints/preferences
- Reviewed existing related code

**After I generate code:**
- You review every line
- You understand how it works
- You test it thoroughly
- You ask me to explain anything unclear
- You integrate it into the system

**During development:**
- Give me feedback on my suggestions
- Tell me what's working/not working
- Ask questions to understand better
- Make decisions when needed

### My Commitment to You

**I will:**
- Generate clean, professional code (no emojis)
- Follow all coding standards in this document
- Write comprehensive tests
- Document everything clearly
- Help you learn and understand
- Support debugging and troubleshooting
- Be available throughout the 14-week project

**I won't:**
- Get frustrated with questions
- Judge your learning curve
- Rush you through understanding
- Make decisions that should be yours
- Leave you with code you don't understand

---

**Let's build something amazing together!** (This is the last emoji, I promise - no emojis in code!)
