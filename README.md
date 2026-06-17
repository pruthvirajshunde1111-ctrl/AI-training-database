# Data Factory

**AI Training Dataset Generation Pipeline** — A modular, production-ready system for transforming raw documents into high-quality AI training datasets. Supports multiple document sources, LLM-powered and rule-based task generation, automated quality control, and interactive monitoring.

![Python](https://img.shields.io/badge/python-3.10%2B-111827?style=flat-square)
![Build](https://img.shields.io/badge/tests-35%2F35-passing-059669?style=flat-square)
![Coverage](https://img.shields.io/badge/license-MIT-6b7280?style=flat-square)

---

## Architecture

```mermaid
graph TB
    subgraph Input["Document Sources"]
        A1[PDF Files]
        A2[Text / Markdown / CSV]
        A3[Web URLs]
    end

    subgraph Loaders["Loader Layer"]
        B1[PDFLoader<br/>PyMuPDF]
        B2[FileLoader<br/>txt / csv / json / md]
        B3[URLLoader<br/>Trafilatura + Decodo API]
    end

    subgraph Processors["Processing Layer"]
        C1[TextCleaner<br/>Normalize / Strip / De-URL]
        C2[TextChunker<br/>Recursive / Fixed / Paragraph / Sentence]
    end

    subgraph Generators["Task Generation Layer"]
        D1[QA Generator<br/>LLM + Rule-based Fallback]
        D2[Summarization<br/>Extractive / Abstractive / Bullet]
        D3[Classification<br/>Topic / Sentiment / Category]
    end

    subgraph Quality["Quality Control Layer"]
        E1[Toxicity Check]
        E2[Bias Detection]
        E3[Diversity Scoring]
        E4[Coherence Analysis]
    end

    subgraph Export["Export Layer"]
        F1[JSON / JSONL Export]
        F2[Streamlit Dashboard]
        F3[CLI Tool]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    B1 --> C1
    B2 --> C1
    B3 --> C1
    C1 --> C2
    C2 --> D1 & D2 & D3
    D1 & D2 & D3 --> E1 & E2 & E3 & E4
    E1 & E2 & E3 & E4 --> F1 & F2 & F3
```

---

## Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI / Dashboard
    participant Factory as DataFactory
    participant Loader as Document Loader
    participant Processor as Text Processor
    participant Generator as Task Generator
    participant QC as Quality Control
    participant Export as Dataset Export

    User->>CLI: Provide sources + tasks
    CLI->>Factory: run(sources, tasks)
    Factory->>Loader: load_sources()
    Loader-->>Factory: List[Document]
    Factory->>Processor: process_documents()
    Processor->>Processor: Clean text
    Processor->>Processor: Chunk text
    Processor-->>Factory: List[TextChunk]
    Factory->>Generator: generate_examples(chunks)

    alt LLM Available
        Generator->>Generator: LLM-based generation
    else Rule-based Fallback
        Generator->>Generator: Keyword / Heuristic extraction
    end

    Generator-->>Factory: List[TaskExample]
    Factory->>QC: evaluate_quality(examples)
    QC->>QC: Toxicity / Bias / Diversity / Coherence
    QC-->>Factory: Passed + Failed examples
    Factory->>Export: assemble_dataset()
    Export-->>Factory: Dataset
    Factory-->>CLI: Dataset summary
    CLI-->>User: Results + Quality Report
```

---

## Features

### Document Ingestion
| Source | Loader | Capabilities |
|--------|--------|-------------|
| PDF | PyMuPDF | Text extraction, preserves structure |
| Text files | Built-in reader | `.txt`, `.md`, `.csv`, `.json`, `.html` |
| Web URLs | Trafilatura + Decodo API | Automatic content extraction, fallback chain |

### Text Processing
- **Cleaning**: Unicode normalization, whitespace stripping, URL/HTML removal, deduplication
- **Chunking**: 4 strategies — recursive (default), fixed-size, paragraph-boundary, sentence-boundary
- **Configurable**: Overlap control, min/max chunk length, language detection

### Task Generation
| Task | LLM Mode | Rule-based Fallback |
|------|----------|-------------------|
| **QA Pairs** | GPT-4o-mini generates factual, inferential, analytical questions | Keyword extraction from sentences, interrogative detection |
| **Summarization** | GPT-4o-mini: extractive + abstractive + bullet points | Top-N sentence extraction, bullet formatting |
| **Classification** | GPT-4o-mini: topic, sentiment, category labels | Keyword-based heuristics, regex patterns |

### Quality Control
| Check | Method | Default Threshold |
|-------|--------|-----------------|
| Toxicity | Pattern-based detection | 0.1 |
| Bias | Gender / race / religious term scoring | 0.2 |
| Diversity | N-gram overlap analysis | 0.15 |
| Coherence | Sentence-logic and flow analysis | 0.3 |
| Overall | Weighted composite score | 0.3 |

### Export Formats
- **JSON** — Full dataset with metadata and quality scores
- **JSONL** — One example per line for streaming / LLM fine-tuning
- **Dashboard** — Interactive preview, filtering, and download

---

## Tech Stack

```mermaid
mindmap
  root((Data Factory))
    Python
      Pydantic v2
      PyMuPDF
      Trafilatura
      Requests
      Regex
    LLM Integration
      OpenRouter API
      OpenAI API
      Auto-provider detection
      Token cost tracking
    Interfaces
      Streamlit Dashboard
      CLI (Click)
      Python SDK
    Quality
      Toxicity scoring
      Bias detection
      Diversity metrics
      Coherence analysis
    Testing
      Pytest (22 unit)
      E2E pipeline tests (35)
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/pruthvirajshunde1111-ctrl/AI-training-database.git
cd AI-training-database
pip install -e .
```

### Configure API Key (Optional)

Create a `.env` file:

```env
DATAFACTORY_LLM_API_KEY=sk-or-v1-your-key-here
```

Without an API key, the pipeline runs in **rule-based fallback mode** — no external API calls, instant results.

### Run the Pipeline

**Via CLI:**

```bash
data-factory run --sources document.pdf --tasks qa summarization
data-factory run --sources https://example.com --tasks classification
data-factory list-templates
data-factory config
```

**Via Python SDK:**

```python
from data_factory import DataFactory

factory = DataFactory()
dataset = factory.run(
    sources=["notes.txt", "https://en.wikipedia.org/wiki/Python_(programming_language)"],
    tasks=["qa", "summarization"],
    max_chunks=5,
)

print(f"Generated {dataset.size} examples")
dataset.export("output/my_dataset.jsonl")
```

**Via Dashboard:**

```bash
streamlit run data_factory/dashboard/app.py
```

Open `http://localhost:8502` in your browser.

---

## Project Structure

```
data_factory/
├── __init__.py              # Public API exports
├── bot.py                   # DataFactory orchestrator
├── config.py                # FactorySettings (Pydantic)
├── models.py                # Data schemas (Document, Dataset, etc.)
│
├── loaders/
│   ├── base.py              # Abstract BaseLoader
│   ├── file_loader.py       # txt / csv / json / md
│   ├── pdf_loader.py        # PyMuPDF integration
│   └── url_loader.py        # Web scraping with fallback chain
│
├── processors/
│   ├── cleaner.py           # 8-step text cleaning pipeline
│   ├── chunker.py           # 4 chunking strategies
│   └── pipeline.py          # Clean → Chunk orchestration
│
├── tasks/
│   ├── templates.py         # Recipe cards for each task
│   ├── manager.py           # Dynamic generator loading
│   ├── qa.py                # QA pair generation
│   ├── summarization.py     # Summarization generation
│   └── classification.py    # Classification generation
│
├── quality/
│   ├── pipeline.py          # Quality orchestration
│   ├── toxicity.py          # Toxicity scoring
│   ├── bias.py              # Bias detection
│   ├── diversity.py         # Diversity analysis
│   └── coherence.py         # Coherence evaluation
│
├── integrations/
│   ├── decodo_api.py        # Decodo scraping API client
│   └── web_scraper.py       # 4-backend fallback chain
│
├── utils/
│   ├── llm_client.py        # Centralized LLM client (OpenAI / OpenRouter)
│   ├── cost_tracker.py      # Token & cost accounting
│   ├── metadata.py          # Run metadata tracking
│   └── logger.py            # Structured logging
│
├── cli/
│   └── main.py              # Click CLI (run, list-templates, config, version)
│
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
│
└── tests/
    ├── test_models.py       # 7 model tests
    ├── test_processors.py   # 8 processor tests
    └── test_quality.py      # 7 quality tests
```

---

## Module Details

### Data Models (`models.py`)

```mermaid
classDiagram
    class Document {
        +str id
        +str source
        +DocumentType source_type
        +str content
        +Dict metadata
    }
    class TextChunk {
        +str id
        +str document_id
        +str text
        +int index
        +int token_count
    }
    class TaskExample {
        +str id
        +TaskType task_type
        +str input
        +str expected_output
        +Dict quality_scores
        +Dict metadata
    }
    class QualityReport {
        +str example_id
        +bool passed
        +Dict scores
        +List issues
    }
    class Dataset {
        +str id
        +str name
        +List~TaskExample~ examples
        +List~QualityReport~ quality_reports
        +Dict metadata
    }
    Document "1" --> "*" TextChunk
    TextChunk "1" --> "*" TaskExample
    Dataset "1" --> "*" TaskExample
    Dataset "1" --> "*" QualityReport
```

### Loader Decision Flow

```mermaid
flowchart LR
    S[Source Input] --> C{Starts with<br/>http:// or https://?}
    C -->|Yes| URL[URLLoader]
    C -->|No| P{Ends with .pdf?}
    P -->|Yes| PDF[PDFLoader]
    P -->|No| FILE[FileLoader]
    URL --> D{Decodo API<br/>configured?}
    D -->|Yes| DC[Decodo Client]
    D -->|No| TF[Trafilatura<br/>Extraction]
    DC -->|Fails| TF
    TF --> DOC[Document]
    PDF --> DOC
    FILE --> DOC
```

### LLM Client Routing

```mermaid
flowchart TD
    KEY[API Key] --> P{Starts with<br/>sk-or-v1-?}
    P -->|Yes| OR[OpenRouter Provider]
    P -->|No| OA[OpenAI Provider]
    OR --> REQ[POST /chat/completions]
    OA --> REQ
    REQ --> R{Response OK?}
    R -->|Yes| PARSE[JSON Parse]
    R -->|No| FALLBACK[Rule-based<br/>Fallback]
    PARSE --> EX[TaskExamples]
    FALLBACK --> EX
```

---

## Testing

```bash
# Unit tests (22 tests)
python -m pytest data_factory/tests/ -v

# End-to-end pipeline tests (35 tests)
python test_full_pipeline.py
```

All 35 end-to-end tests validate:
- Document loading (files, PDFs, URLs)
- Text cleaning and chunking
- LLM-based and rule-based task generation
- Quality control evaluation
- Dataset assembly and export
- Cost tracking

---

## Dashboard

The Streamlit dashboard provides four tabs for interactive monitoring:

| Tab | Purpose |
|-----|---------|
| **Pipeline** | Configure sources, tasks, and run the pipeline with progress tracking |
| **Dataset** | Preview generated examples, filter by task type and quality status, download as JSON or JSONL |
| **Quality Report** | Visual quality metrics with per-example breakdown and flagged issues |
| **Run History** | Timeline of all pipeline runs with key stats |

Run with:

```bash
streamlit run data_factory/dashboard/app.py
```

Then open `http://localhost:8502`.

---

## Configuration

Key settings available via environment variables (prefix `DATAFACTORY_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATAFACTORY_LLM_API_KEY` | — | OpenRouter or OpenAI API key |
| `DATAFACTORY_LLM_MODEL` | `gpt-4o-mini` | Model for LLM generation |
| `DATAFACTORY_CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `DATAFACTORY_CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `DATAFACTORY_LOG_LEVEL` | `INFO` | Logging verbosity |
| `DATAFACTORY_OUTPUT_DIR` | `output` | Dataset export directory |

Settings can also be passed programmatically:

```python
from data_factory.config import FactorySettings

settings = FactorySettings(
    llm_api_key="sk-...",
    chunk_size=1024,
    enable_toxicity_check=True,
)
```

---

## License

MIT
