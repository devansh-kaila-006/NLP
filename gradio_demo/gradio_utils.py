"""
Helper utilities for Multi-Modal RAG Gradio Demo

Provides formatting and processing functions for the UI.
"""

import re
from typing import Dict, Any, List


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to readable MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (MM:SS)
    """
    if not seconds or seconds <= 0:
        return "0:00"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def create_youtube_timestamp_url(video_id: str, timestamp: float) -> str:
    """
    Create a clickable YouTube URL with timestamp.

    Args:
        video_id: YouTube video ID
        timestamp: Timestamp in seconds

    Returns:
        Complete YouTube URL with timestamp parameter
    """
    if not video_id:
        return "#"

    timestamp_int = int(timestamp)
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp_int}"


def calculate_confidence_color(score: float) -> str:
    """
    Calculate color for confidence indicators.

    Args:
        score: Confidence score (0-1)

    Returns:
        CSS color name
    """
    if score >= 0.7:
        return "green"
    elif score >= 0.4:
        return "orange"
    else:
        return "red"


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def format_modality_icon(modality: str) -> str:
    """
    Get emoji icon for modality type.

    Args:
        modality: Modality type (pdf, video, aman)

    Returns:
        Emoji icon
    """
    icons = {
        'pdf': '📄',
        'video': '🎥',
        'aman': '🤖'
    }
    return icons.get(modality.lower(), '📄')


def format_source_badge(modality: str, score: float = None) -> str:
    """
    Create a formatted badge for source modality.

    Args:
        modality: Modality type
        score: Optional relevance score

    Returns:
        HTML badge string
    """
    icon = format_modality_icon(modality)
    color = calculate_confidence_color(score) if score else "gray"

    if score:
        return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.9em;">{icon} {modality.upper()} ({score:.2f})</span>'
    else:
        return f'<span style="background-color: gray; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.9em;">{icon} {modality.upper()}</span>'


def format_progress_bar(score: float, width: str = "100px") -> str:
    """
    Create a visual progress bar for confidence scores.

    Args:
        score: Confidence score (0-1)
        width: Bar width

    Returns:
        HTML progress bar
    """
    percentage = score * 100
    color = calculate_confidence_color(score)

    return f'''
    <div style="width: {width}; background-color: #e0e0e0; border-radius: 4px; overflow: hidden;">
        <div style="width: {percentage}%; background-color: {color}; height: 8px; border-radius: 4px;"></div>
    </div>
    '''


def format_modality_section(modality: str, sources: List[Dict[str, Any]]) -> str:
    """
    Format sources section for a specific modality.

    Args:
        modality: Modality type
        sources: List of sources

    Returns:
        Formatted HTML section
    """
    if not sources:
        return ""

    icon = format_modality_icon(modality)
    html_parts = [f"<h4>{icon} {modality.upper()} Sources ({len(sources)})</h4>"]

    for i, source in enumerate(sources, 1):
        score = source.get('score', source.get('relevance', 0))
        text = source.get('text', '')
        truncated_text = truncate_text(text, 150)

        html_parts.append(f"""
        <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid {calculate_confidence_color(score)}; background-color: #f9f9f9;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <strong>Source {i}</strong>
                {format_source_badge(modality, score)}
            </div>
        """)

        # Add modality-specific information
        if modality == 'video':
            timestamp = format_timestamp(source.get('timestamp_start', 0))
            video_url = source.get('video_url', '')
            video_title = source.get('video_title', 'Unknown Video')

            html_parts.append(f"""
            <div style="margin: 5px 0;">
                <strong>🎬 {video_title}</strong><br>
                <strong>⏱️ Timestamp:</strong> {timestamp}
                {f'<br><strong>🔗</strong> <a href="{video_url}" target="_blank">Watch on YouTube</a>' if video_url else ''}
            </div>
            """)

        elif modality == 'pdf':
            chapter = source.get('chapter', 'Unknown Chapter')
            section = source.get('section', 'Unknown Section')

            html_parts.append(f"""
            <div style="margin: 5px 0;">
                <strong>📖 {chapter}</strong><br>
                <strong>📑 Section:</strong> {section}
            </div>
            """)

        elif modality == 'aman':
            category = source.get('category', 'AI')
            url = source.get('url', '')

            html_parts.append(f"""
            <div style="margin: 5px 0;">
                <strong>🏷️ Category:</strong> {category}
                {f'<br><strong>🔗</strong> <a href="{url}" target="_blank">Read on Aman.ai</a>' if url else ''}
            </div>
            """)

        # Add text preview
        html_parts.append(f"""
        <div style="margin-top: 8px; font-style: italic; color: #666;">
            "{truncated_text}"
        </div>
        </div>
        """)

    return "".join(html_parts)


def format_timings_breakdown(timings: Dict[str, float]) -> str:
    """
    Format timing breakdown for display.

    Args:
        timings: Dictionary of timing components

    Returns:
        Formatted HTML timing breakdown
    """
    if not timings:
        return "<p>No timing data available</p>"

    total = timings.get('total', sum(timings.values()))

    html_parts = ["<h4>⏱️ Processing Breakdown</h4><ul>"]

    # Define display names and order
    timing_components = [
        ('Modality Prediction', 'modality_prediction'),
        ('Retrieval', 'retrieval'),
        ('Reranking', 'reranking'),
        ('Generation', 'generation'),
        ('Temporal Coherence', 'temporal_coherence')
    ]

    for display_name, key in timing_components:
        if key in timings:
            time_value = timings[key]
            percentage = (time_value / total * 100) if total > 0 else 0
            html_parts.append(f"<li><strong>{display_name}:</strong> {time_value:.2f}s ({percentage:.1f}%)</li>")

    html_parts.append("</ul>")
    return "".join(html_parts)


def create_example_queries() -> Dict[str, List[str]]:
    """
    Create organized example queries by domain.

    Returns:
        Dictionary of domains and their example queries
    """
    return {
        "Machine Learning Fundamentals": [
            "What is linear regression in machine learning?",
            "Explain the concept of overfitting and underfitting",
            "What is the difference between supervised and unsupervised learning?",
            "How does gradient descent optimization work?"
        ],
        "Deep Learning Architectures": [
            "Explain the architecture of convolutional neural networks",
            "What are the key differences between RNNs and LSTMs?",
            "How does backpropagation work in neural networks?",
            "What is the vanishing gradient problem?"
        ],
        "Natural Language Processing": [
            "What is transformer architecture in NLP?",
            "Explain attention mechanism and self-attention",
            "How does word embedding work in NLP?",
            "What are the main applications of BERT and GPT models?"
        ],
        "Computer Vision": [
            "What is image segmentation and how does it work?",
            "Explain object detection algorithms like YOLO",
            "What are convolutional neural networks used for in computer vision?",
            "How does transfer learning work for image classification?"
        ],
        "Modern AI Topics": [
            "What is prompt engineering in large language models?",
            "Explain the concept of few-shot learning",
            "What are the main challenges in AGI development?",
            "How do reinforcement learning algorithms work?"
        ]
    }


def format_system_stats(stats: Dict[str, Any]) -> str:
    """
    Format system statistics for display.

    Args:
        stats: System statistics dictionary

    Returns:
        Formatted HTML statistics section
    """
    html_parts = ["<h4>📊 System Statistics</h4><ul>"]

    if stats.get('total_queries', 0) > 0:
        html_parts.append(f"<li><strong>Queries Processed:</strong> {stats['total_queries']}</li>")
        html_parts.append(f"<li><strong>Cache Hit Rate:</strong> {stats.get('cache_hit_rate', 0):.1%}</li>")
        html_parts.append(f"<li><strong>Average Query Time:</strong> {stats.get('avg_query_time', 0):.2f}s</li>")

    html_parts.append("</ul>")
    return "".join(html_parts)