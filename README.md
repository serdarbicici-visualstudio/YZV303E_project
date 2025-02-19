# YZV303E Deep Learning  
## Project Report  

### Evaluating Retrieval-Augmented Generation Techniques in the Tourism Domain of Istanbul  

**Authors:**  
- Batuhan Sal - 150210316 (sal21@itu.edu.tr)  
- Ömer Erdağ - 150210332 (erdag21@itu.edu.tr)  
- Serdar Biçici - 150210331 (bicici21@itu.edu.tr)  

---

## Table of Contents  
1. [Problem Description](#problem-description)  
2. [Literature Review](#literature-review)  
   - [Retrieval-Augmented Generation](#retrieval-augmented-generation)  
   - [Domain-Specific Enhancements](#domain-specific-enhancements)  
   - [Best Practices in RAG](#best-practices-in-rag)  
3. [Dataset](#dataset)  
   - [Data Sources](#data-sources)  
   - [Q&A Generation Using Qwen-70B Model](#qa-generation-using-qwen-70b-model)  
   - [Final Dataset](#final-dataset)  
4. [Method Formulation](#method-formulation)  
   - [Chunking Methods](#chunking-methods)  
   - [Vector Databases](#vector-databases)  
   - [Retrieval Methods](#retrieval-methods)  
   - [Reranking Methods](#reranking-methods)  
   - [Document Repacking](#document-repacking)  
   - [Summarization](#summarization)  
5. [Real-world Application](#real-world-application)  
6. [Experimental Evaluation](#experimental-evaluation)  
7. [Conclusion](#conclusion)  
8. [References](#references)  

---

## Problem Description  

Large Language Models (LLMs) demonstrate exceptional capabilities across various fields. However, their accuracy diminishes in specialized domains, such as tourism, due to outdated or inaccurate information. This issue is particularly critical for tourists who require reliable, up-to-date guidance.  

This project addresses these challenges by focusing on **Istanbul's tourism sector** using **Retrieval-Augmented Generation (RAG)** techniques. A specialized dataset of Istanbul's tourist attractions is compiled, and various RAG methods are implemented to generate and refine **domain-specific Q&A pairs**.  

The goal is to develop a **robust question-answering system** that delivers **high-quality, fact-checked** information to tourists and locals, demonstrating RAG's potential in **enhancing the reliability of domain-specific retrieval systems**.  

---

## Literature Review  

### Retrieval-Augmented Generation  

Retrieval-Augmented Generation (RAG) improves LLMs by integrating **external knowledge sources** to enhance accuracy and reduce hallucinations.  

Gao et al. (2023) categorized RAG into three paradigms:  
- **Naive RAG**  
- **Advanced RAG**  
- **Modular RAG**  

Each paradigm incorporates retrieval mechanisms at different stages to **mitigate outdated knowledge and factual inaccuracies**.  

### Domain-Specific Enhancements  

Wei et al. (2024) introduced **TourLLM**, a fine-tuned model on the **Cultour dataset**. Their results show that **Supervised Fine-Tuning (SFT)** enhances the accuracy and relevance of travel-related content.  

### Best Practices in RAG  

Wang et al. (2024) emphasized the **importance of multimodal retrieval capabilities** and proposed strategies to optimize query-dependent retrievals, improving domain-specific **LLM applications**.  

---

## Dataset  

### Data Sources  

The dataset is built using multiple sources:  

- **Istanbul Attractions List** (from Kaggle)  
- **Official Tourism Data** (scraped from *muze.gen.tr*)  
- **Wikipedia API** (structured and semi-structured historical data)  
- **Municipal Resources** (Atatürk Library, IBB Municipality)  
- **Community Forums** (*Rick Steves' travel forum* for real user questions)  
- **Generated Q&A Pairs** (using **Qwen-70B** language model)  

The collected data is consolidated into a **unified knowledge base** for RAG experiments.  

### Q&A Generation Using Qwen-70B Model  

1. **Prompt Engineering:** Crafted structured prompts ensuring factual, diverse, and educational questions.  
2. **Contextual Input:** Segmented knowledge base using **LLamaParser** to structure input data.  
3. **Synthetic Answer Generation:** Generated and **validated** answers using **Gemma2 9B model**.  
4. **Evaluation:**  
   - **Faithfulness** (accuracy with knowledge base)  
   - **Contextual Relevance** (alignment with context)  
   - **Question Diversity** (varied question formats)  
5. **Manual Review:** Filtered redundant, irrelevant, or ambiguous Q&A pairs.  

### Final Dataset  

The dataset consists of:  
- **Curated Q&A pairs** from community forums.  
- **Synthetic Q&A pairs** generated using **Qwen-70B**.  

---

## Method Formulation  

### Chunking Methods  

1. **Fixed-size Chunking:** Divides text into equal-sized chunks (efficient but may disrupt context).  
2. **Recursive Chunking:** Uses **NLP techniques** (e.g., **SpaCy, NLTK**) for context-aware segmentation.  
3. **Semantic Chunking:** Clusters semantically similar sentences for **contextually coherent** chunks.  

### Vector Databases  

- **FAISS:** Lightweight, used for initial experiments.  
- **Milvus:** More **scalable**, supports **hybrid search** for better retrieval accuracy.  

### Retrieval Methods  

1. **Query2Doc:** Expands user queries into **hypothetical documents** for better retrieval.  
2. **HyDE:** A simplified version of Query2Doc, generating a **hypothetical query representation**.  
3. **Hybrid Search:** Combines **BM25 (sparse)** and **dense vector retrieval** for improved accuracy.  

### Reranking Methods  

1. **BGE Reranker:** Uses dense embeddings to refine retrieval ranking.  
2. **TildeV2 Model:** Balances lexical and semantic retrieval, optimizing relevance.  

### Document Repacking  

Rearranges retrieved documents for **optimal information flow**:  
- **Forward Packing:** Retains original ranking order.  
- **Reverse Packing:** Presents lower-ranked documents first.  
- **Side Packing:** Mixes high- and low-ranked results.  

### Summarization  

Condenses retrieved chunks into **concise summaries** for **efficient user interaction**.  

---

## Real-World Application  

A **web-based question-answering system** for Istanbul's tourism sector:  
- Users input queries (e.g., **"What are the visiting hours for Topkapı Palace?"**)  
- The RAG system retrieves **fact-checked** answers from the knowledge base.  
- Future expansion: Supporting **multiple cities** or **domains**.  

---

## Experimental Evaluation  

### Key Evaluation Metrics  
- **Faithfulness:** Ensures generated responses align with retrieved knowledge.  
- **Relevancy:** Measures context alignment.  
- **Factual Correctness:** Evaluates the accuracy of generated answers.  

### Experimental Findings  

- **Semantic chunking** outperforms **fixed and recursive** chunking.  
- **Hybrid retrieval** (BM25 + dense vectors) improves accuracy.  
- **RAG significantly enhances factual correctness** over standard LLM responses.  

### Final Comparison (RAG vs No RAG)  

| Method    | Faithfulness | Relevancy | Factual Correctness | Overall |
|-----------|-------------|-----------|---------------------|---------|
| No RAG   | 0.49        | 0.81      | 0.49                | 0.65    |
| RAG      | **0.72**    | **0.85**  | **0.72**            | **0.79**  |

---

## Conclusion  

This project successfully demonstrated the **effectiveness of Retrieval-Augmented Generation (RAG)** in the tourism sector. Key takeaways:  
- **RAG reduces hallucinations & improves factual correctness**.  
- **Optimized chunking, retrieval, and reranking improve accuracy**.  
- **Future work:** Expanding the **knowledge base** and integrating **more retrieval techniques**.  

---

## References  

1. Gao et al. (2023). *Retrieval-Augmented Generation for LLMs: A Survey*.  
2. Wei et al. (2024). *TourLLM: Enhancing LLMs with Tourism Knowledge*.  
3. Wang et al. (2024). *Best Practices in Retrieval-Augmented Generation*.  
