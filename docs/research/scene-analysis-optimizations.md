# Scene Analysis Optimization Strategies

This document outlines research and potential strategies for optimizing the scene analysis pipeline to achieve lower latency, reduce cost, and enable faster, more responsive gameplay interactions.

Our performance benchmarks have established that our current VLM-only pipeline has a hard latency floor of approximately **2-3 seconds**. While acceptable for general context awareness, this is too slow for critical, real-time events like player death or low health warnings. 

To overcome this, we should adopt a **hybrid analysis model** and explore specialized, high-performance VLM alternatives.

## 1. Hybrid Analysis: Fast-Path Heuristics + Slow-Path VLM

The most impactful short-term strategy is to stop relying on the VLM for everything. We can implement a fast-path using simple, local computer vision (CV) heuristics that run on every frame, reserving the expensive VLM for deeper analysis at a slower cadence.

-   **Fast-Path (Local CV Heuristics)**:
    -   **Purpose**: Sub-second detection of critical, predefined events.
    -   **Implementation**: Lightweight Python scripts using libraries like OpenCV or even just Pillow for pixel analysis.
    -   **Examples**:
        -   **Health Bar Monitor**: Continuously check a small, predefined screen region for the health bar's color. If it turns red (`#CRITICAL_HEALTH_COLOR`), trigger an immediate low-health event.
        -   **Death Screen Detector**: Use template matching to look for the "YOU DIED" text or a specific visual signature that appears on the death screen.
        -   **Menu Detector**: Check for a consistent UI element or color shift that indicates a menu is open, allowing us to pause analysis.

-   **Slow-Path (VLM Analysis)**:
    -   **Purpose**: Detailed scene understanding, context gathering, and answering complex user questions.
    -   **Implementation**: Continue using `gemini-2.5-flash-lite` at a 1-3 second interval, as benchmarked.

This hybrid model provides the best of both worlds: near-instantaneous reaction for critical events and rich, contextual understanding for everything else.

## 2. Alternative VLM Models

Our research identified several promising VLM models that are either open-source or specifically designed for UI understanding and performance. If the latency of our current Gemini model proves to be a bottleneck even for the slow-path, we should evaluate these alternatives.

-   **FastVLM (Apple)**: An open-source model explicitly designed for efficient vision encoding and low-latency, real-time applications. Its public demos show impressive performance, making it a top candidate for evaluation.
-   **ScreenAI (Google)**: A 5B parameter model specializing in UI and infographics. Its training on screen annotations is highly relevant to our use case and could yield better accuracy and speed for UI-specific tasks.

## 3. Future Model Integration (e.g., GPT-5 Nano)

As the AI landscape evolves, new generations of small, powerful models will become available. Competitors like Questie.ai already mention `GPT-5 Nano`. We should remain agile and be prepared to integrate these next-gen models as they are released, as they will likely offer superior performance-per-watt and lower latency.

## 4. Custom VLM Fine-Tuning

This is the ultimate long-term strategy for achieving peak performance and accuracy. By fine-tuning a base VLM (like FastVLM or another open-source model) on a large, curated dataset of Tunic screenshots, we can create a highly specialized model that is:

-   **Faster**: It only needs to know about Tunic's specific UI, enemies, and environments, reducing the computational overhead.
-   **More Accurate**: It can be trained to recognize game-specific nuances that a general-purpose model might miss.
-   **Cheaper**: Running a smaller, fine-tuned model (especially on local hardware or a private cloud) can be more cost-effective than making API calls to a large commercial model.

Competitors like Hakko.ai with their proprietary `LynkSoul VLM v1` are likely already doing this. Building our own dataset and fine-tuning pipeline would be a significant engineering effort but would provide a powerful competitive advantage.
