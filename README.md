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

### 1. Intelligent Intent Routing
The core logic resides in a rule-based classifier that analyzes the linguistic structure of the user's question:
* **Technical/Factual:** Questions regarding dimensions, hardware, or compatibility.
* **Subjective/Opinionated:** Questions regarding user satisfaction, quality, or reliability.
* **Hybrid Analysis:** Simultaneous execution for complex queries requiring both data points.

### 2. Deep Learning Stack
| Component | Model | Functional Utility |
| :--- | :--- | :--- |
| **Extractive QA** | `deepset/bert-base-cased-squad2` | High-precision extraction of facts from unstructured product descriptions. |
| **Sentiment Analysis**| `distilbert-base-uncased-finetuned-sst-2` | Statistical aggregation of customer sentiment across 50+ real-time reviews. |

---

## 🛠️ Tech Stack
* **Core Logic:** Python, Flask (RESTful API)
* **Deep Learning:** Transformers (Hugging Face), PyTorch, BERT
* **Automation:** Playwright (Chromium), BeautifulSoup4
* **Reliability:** Docker, SQLite (Query Persistence), UptimeRobot

---

## 🚀 Installation & Local Development

### Prerequisites
* Python 3.9+
* Docker (optional)

### Setup
```bash
# Clone the repository
git clone [https://github.com/Aryan2659/ecom-qa-bert-v2](https://github.com/Aryan2659/ecom-qa-bert-v2) && cd ecom-qa-bert-v2

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
python -m playwright install chromium

# Launch the Application
python -m src.app
