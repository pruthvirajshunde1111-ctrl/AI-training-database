"""Single comprehensive test case for the full Data Factory pipeline.

Tests: document loading, text processing, task generation (LLM + rule-based),
quality control, dataset assembly, and export.

Usage: python test_full_pipeline.py
"""

import json, tempfile, os
from pathlib import Path

os.environ["DATAFACTORY_LOG_LEVEL"] = "WARNING"

from data_factory import DataFactory, FactorySettings
from data_factory.models import Document, TaskExample, TaskType, Dataset
from data_factory.loaders import FileLoader
from data_factory.processors import TextCleaner, TextChunker, ProcessingPipeline
from data_factory.quality import ToxicityChecker, BiasDetector, CoherenceEvaluator, QualityPipeline


def test_full_pipeline():
    """Run one complete end-to-end test of the Data Factory."""
    passed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))

    print("=" * 60)
    print("DATA FACTORY - FULL FUNCTIONALITY TEST")
    print("=" * 60)

    # ── 1. MODELS ────────────────────────────────────────────────────
    print("\n--- Models ---")
    doc = Document(source="test.txt", source_type="text", content="Hello world. This is a test.")
    check("Document creation", doc.id.startswith("doc_"))
    doc.compute_word_count()
    check("Word count", doc.word_count == 6)

    chunk = doc  # reuse
    from data_factory.models import TextChunk
    tc = TextChunk(document_id=doc.id, index=0, text="Sample chunk text for testing")
    check("TextChunk creation", tc.id.startswith("chunk_"))

    ex = TaskExample(task_type=TaskType.QA, task_name="qa", input="Q?", expected_output="A")
    check("TaskExample creation", ex.id.startswith("ex_"))

    ds = Dataset(name="test_ds")
    ds.add_example(ex)
    check("Dataset add & size", ds.size == 1)
    check("To dict has task_type", ex.to_dict()["task_type"] == "qa")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        ds.export(str(path))
        check("Dataset export JSONL", path.exists() and path.stat().st_size > 0)

    # ── 2. LOADERS ────────────────────────────────────────────────────
    print("\n--- Loaders ---")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Machine learning is transforming how computers understand data. Deep learning uses neural networks. Natural language processing enables text understanding.")
        txt_path = f.name

    loader = FileLoader()
    docs = loader.load(txt_path)
    check("FileLoader loads .txt", len(docs) == 1)
    check("Document has content", len(docs[0].content) > 50)
    Path(txt_path).unlink(missing_ok=True)

    # ── 3. PROCESSORS ────────────────────────────────────────────────
    print("\n--- Processors ---")
    cleaner = TextCleaner()
    cleaned = cleaner.clean("  Hello   <b>world</b>  from https://example.com  ")
    check("TextCleaner strips HTML", "<b>" not in cleaned)
    check("TextCleaner strips URL", "https://" not in cleaned)
    check("TextCleaner normalizes whitespace", "  " not in cleaned)

    chunker = TextChunker(chunk_size=50, strategy="sentence", min_chunk_length=5)
    chunks = chunker.chunk("First sentence here. Second sentence here. Third sentence here.", document_id="doc_1")
    check("Chunker splits sentences", len(chunks) >= 2)

    pipe = ProcessingPipeline()
    test_doc = Document(source="test.txt", source_type="text",
                        content="Artificial intelligence AI is transforming industries. Machine learning finds patterns in data. Deep learning uses neural networks.")
    p_chunks = pipe.process(test_doc)
    check("Pipeline processes document", len(p_chunks) > 0)

    # ── 4. TASK GENERATORS (rule-based fallback) ─────────────────────
    print("\n--- Task Generation (rule-based) ---")
    from data_factory.tasks import QAGenerator, SummarizationGenerator, ClassificationGenerator

    qa_gen = QAGenerator()
    qa_examples = qa_gen.generate(p_chunks, max_examples=3)
    check("QA generator produces examples", len(qa_examples) > 0)

    summ_gen = SummarizationGenerator()
    summ_examples = summ_gen.generate(p_chunks)
    check("Summarization generator produces examples", len(summ_examples) > 0)

    cls_gen = ClassificationGenerator()
    cls_examples = cls_gen.generate(p_chunks)
    check("Classification generator produces examples", len(cls_examples) > 0)

    # ── 5. QUALITY CONTROL ───────────────────────────────────────────
    print("\n--- Quality Control ---")
    tox = ToxicityChecker()
    clean_ex = TaskExample(task_type=TaskType.QA, task_name="qa", input="What is Python?", expected_output="A language")
    r1 = tox.check(clean_ex)
    check("Toxicity: clean text passes", r1.passed)

    toxic_ex = TaskExample(task_type=TaskType.QA, task_name="qa", input="You are an idiot", expected_output="Shut up")
    r2 = tox.check(toxic_ex)
    check("Toxicity: toxic text flagged", not r2.passed)

    bias = BiasDetector()
    r3 = bias.check(clean_ex)
    check("Bias: clean text OK", "bias" in r3.scores)

    coherence = CoherenceEvaluator()
    r4 = coherence.check(clean_ex)
    check("Coherence: score computed", r4.scores["coherence"] > 0)

    quality_pipe = QualityPipeline()
    r5 = quality_pipe.evaluate(clean_ex)
    check("QualityPipeline: all metrics present",
          all(m in r5.scores for m in ["toxicity", "bias", "diversity", "coherence", "overall"]))

    # ── 6. FULL PIPELINE (rule-based, no LLM) ────────────────────────
    print("\n--- Full Pipeline (rule-based) ---")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Artificial intelligence is transforming how computers understand and process information. "
                "Machine learning algorithms can detect patterns in data that humans might miss. "
                "Deep learning uses multi-layered neural networks to solve complex problems. "
                "Natural language processing enables computers to understand human language. "
                "Computer vision allows machines to interpret visual information from the world.")
        src_path = f.name

    settings = FactorySettings(
        chunk_size=256,
        llm_api_key="",  # empty string = no key
        enable_toxicity_check=True,
        enable_bias_check=True,
        enable_diversity_check=True,
        enable_coherence_check=True,
        log_level="WARNING",
    )
    # Override the env-loading in bot.py so it doesn't re-pick up .env
    import os as _os
    _saved = _os.environ.pop("DATAFACTORY_LLM_API_KEY", None)
    _saved2 = _os.environ.pop("OPENAI_API_KEY", None)
    factory = DataFactory(settings)
    check("Factory initialized without LLM", factory.llm_client is None)
    if _saved:
        _os.environ["DATAFACTORY_LLM_API_KEY"] = _saved
    if _saved2:
        _os.environ["OPENAI_API_KEY"] = _saved2

    dataset = factory.run(sources=[src_path], tasks=["qa", "summarization", "classification"],
                          run_name="full-test", skip_quality=False)
    check("Pipeline runs end-to-end", dataset is not None)
    check("Dataset has examples", dataset.size > 0)

    for ex in dataset.examples:
        check(f"  Example {ex.id[:8]}: valid {ex.task_type.value}",
              bool(ex.input) and bool(ex.expected_output))

    Path(src_path).unlink(missing_ok=True)

    # ── 7. FULL PIPELINE (with LLM via OpenRouter) ───────────────────
    print("\n--- Full Pipeline (with LLM via OpenRouter) ---")
    has_llm = bool(os.environ.get("DATAFACTORY_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if has_llm:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Climate change is one of the most pressing challenges of our time. "
                    "Rising global temperatures are causing extreme weather events. "
                    "Scientists emphasize the need for immediate action to reduce carbon emissions. "
                    "Renewable energy sources like solar and wind power offer sustainable alternatives. "
                    "International cooperation is essential for effective climate policy.")
            llm_path = f.name

        llm_settings = FactorySettings(log_level="WARNING")
        llm_factory = DataFactory(llm_settings)
        check("Factory has LLM client", llm_factory.llm_client is not None)

        if llm_factory.llm_client:
            test_msg = llm_factory.llm_client.chat(
                messages=[{"role": "user", "content": "Reply with just the lowercase word: ok"}],
                max_tokens=10, temperature=0.0
            )
            check("LLM API responds", test_msg is not None and "ok" in test_msg.strip().lower(),
                  f"got '{test_msg.strip() if test_msg else 'None'}'")

            llm_dataset = llm_factory.run(
                sources=[llm_path],
                tasks=["qa", "summarization", "classification"],
                run_name="llm-test", skip_quality=False
            )
            check("LLM pipeline generates examples", llm_dataset.size > 0)

            if llm_dataset.examples:
                for ex in llm_dataset.examples[:2]:
                    check(f"  LLM {ex.task_type.value}: non-empty output",
                          len(ex.expected_output) > 20)

        Path(llm_path).unlink(missing_ok=True)
    else:
        print("  [SKIP] No API key found for LLM test")

    # ── RESULTS ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*60}")
    return passed == total


if __name__ == "__main__":
    success = test_full_pipeline()
    exit(0 if success else 1)
