# Multi-Modal RAG System - Gradio Demo

Professional web-based UI for the Multi-Modal RAG System showcasing:
- 🎯 97% cross-modal prediction accuracy
- ⏱️ 100% temporal coherence
- 🔗 Timestamp-aware video RAG
- ⚡ Sub-4.0s query performance with 400x+ caching

## Features

### Interactive Query Interface
- Clean text input with example queries
- Advanced options for top-K and reranking parameters
- Force modality selection for testing

### Rich Results Display
- **📝 Answer Section**: Professional formatted responses
- **🎯 Modality Prediction**: Visual confidence indicators
- **📚 Sources & Citations**: Clickable video timestamp links
- **⚡ Performance Metrics**: Real-time query breakdown

### System Capabilities
- 12,717 total chunks (PDF: 9,661, Video: 2,923, Web: 133)
- Multi-modal content integration
- Intelligent caching for 400x+ speedup on repeated queries

## Installation

```bash
# Install Gradio dependency
pip install gradio>=4.0.0

# Or install all requirements
pip install -r requirements.txt
```

## Usage

### Local Development
```bash
# From project root
python gradio_demo/gradio_demo_app.py
```

The UI will be available at: `http://localhost:7860`

### Public Sharing
```bash
# Edit gradio_demo/gradio_demo_app.py
# Change share=False to share=True in the launch() call

python gradio_demo/gradio_demo_app.py
```

This will generate a public URL for sharing your demo.

## Demo Queries

Organized by domain:

### Machine Learning Fundamentals
- What is linear regression in machine learning?
- Explain the concept of overfitting and underfitting
- What is the difference between supervised and unsupervised learning?
- How does gradient descent optimization work?

### Deep Learning Architectures
- Explain the architecture of convolutional neural networks
- What are the key differences between RNNs and LSTMs?
- How does backpropagation work in neural networks?
- What is the vanishing gradient problem?

### Natural Language Processing
- What is transformer architecture in NLP?
- Explain attention mechanism and self-attention
- How does word embedding work in NLP?
- What are the main applications of BERT and GPT models?

### Computer Vision
- What is image segmentation and how does it work?
- Explain object detection algorithms like YOLO
- What are convolutional neural networks used for in computer vision?
- How does transfer learning work for image classification?

### Modern AI Topics
- What is prompt engineering in large language models?
- Explain the concept of few-shot learning
- What are the main challenges in AGI development?
- How do reinforcement learning algorithms work?

## Performance Expectations

- **First Query**: ~3-4 seconds (model loading + processing)
- **Cached Queries**: ~0.05 seconds (400x+ speedup)
- **Modality Prediction**: <0.01 seconds (instant)
- **Multi-Modal Retrieval**: ~1.5 seconds (parallel processing)

## System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Internet connection (for LLM API and video links)

## Troubleshooting

### Pipeline Loading Issues
If the pipeline fails to load:
1. Ensure all data files exist in `data/processed/`
2. Check that API keys are set in `.env` file
3. Verify model downloads completed successfully

### Performance Issues
If queries are slow:
1. Check if caching is enabled
2. Verify model preloading completed
3. Monitor system memory usage

### UI Display Issues
If UI elements don't display correctly:
1. Clear browser cache and reload
2. Try a different browser (Chrome, Firefox, Edge)
3. Check browser console for errors

## File Structure

```
gradio_demo/
├── gradio_demo_app.py       # Main application
├── gradio_utils.py          # Helper functions
└── README.md               # This file
```

## Academic Presentation Tips

### Recommended Demo Flow
1. **Introduction**: Start with "What is deep learning?" to show multi-modal sources
2. **Video RAG**: Ask "Explain gradient descent" to highlight temporal coherence
3. **PDF Math**: Try "Derive backpropagation" for mathematical content
4. **Modern AI**: Use "What is prompt engineering?" for Aman.ai content
5. **Performance**: Repeat a query to demonstrate 400x+ caching speedup

### Key Features to Highlight
- **Cross-Modal Prediction**: Show confidence bars and modality reasoning
- **Timestamp Links**: Click video sources to jump to exact moments
- **Performance Metrics**: Point out sub-4.0s query times and cache status
- **Source Diversity**: Note the mix of PDF, video, and web sources

## Support

For issues or questions:
- Check system logs in `logs/rag_pipeline.log`
- Review performance metrics in the UI dashboard
- Consult main documentation in project README

---

**Multi-Modal RAG System v3.0** - Production Release  
**Status**: Demo-Ready ✅  
**Last Updated**: 2026-05-17
