# 🛍️ E-Commerce Intelligence Engine: BERT QA & Sentiment
**A Production-Grade NLP Pipeline for Real-Time Product Analytics**

[![Live Demo](https://img.shields.io/badge/Demo-HuggingFace-blue)](https://huggingface.co/spaces/rnyx/ecom-qa-bert-v2) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview
The **E-Commerce Intelligence Engine** is an end-to-end NLP system designed to bridge the gap between massive retail datasets and consumer queries. By combining real-time web automation with transformer-based deep learning, the engine parses product pages to provide instant, factual answers and aggregated sentiment analysis.

### 🌟 Key Engineering Highlights
* **Dual-Model Orchestration:** Implements an **Intent Router** that classifies incoming queries to dispatch them to the optimal model: extractive **BERT** for technical specs or **DistilBERT** for subjective reviews.
* **Dynamic Web Automation:** Utilizes **Playwright** (Headless Chromium) to render JavaScript-heavy e-commerce environments, ensuring the capture of reviews and specifications that traditional scrapers miss.
* **Production-Ready Resilience:** Features a custom **Failover Layer**; if cloud-based requests face bot-detection (e.g., on Amazon), the system automatically triggers a stealth-fallback mechanism to maintain data flow.
* **DevOps & Containerization:** Fully Dockerized with model weights pre-baked into the image to eliminate runtime latency and ensure high availability on the **Hugging Face Spaces** infrastructure.

---

## 🏗️ System Architecture
The engine follows a modular pipeline designed for low-latency inference and high data integrity:

```text
                        ┌─────────────────────┐
        Question ──────▶│    Intent Router    │ ───┐
                        │   (keyword-based)   │    │
                        └─────────────────────┘    │
                                                   │
                ┌──── "spec" ───┐  "both"    ┌──── "review" ───┐
                │               │    │       │                 │
                ▼               ▼    ▼       ▼                 ▼
        ┌──────────────┐       (both run)          ┌────────────────────┐
        │   BERT QA    │                           │   DistilBERT-SST   │
        │ (extractive) │                           │    (sentiment)     │
        │ from specs + │                           │  batched over all  │
        │ features     │                           │  scraped reviews   │
        └──────────────┘                           └────────────────────┘
              │                                              │
              ▼                                              ▼
      Answer span +                                  Overall verdict +
      confidence +                                   pos/neg % + top 3
      token breakdown                                positive + top 3 negative
