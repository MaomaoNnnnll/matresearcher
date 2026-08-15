"""Quick import verification for the project venv."""
import sys
sys.path.insert(0, "F:/projects/matresearcher/src")

total = 0
passed = 0
failed = []

checks = [
    ("state", "WorkflowState"),
    ("models.literature", "Literature"),
    ("models.knowledge", "KnowledgeRecord"),
    ("models.gap", "ResearchGap"),
    ("tools.llm", "LLMClient"),
    ("tools.sciverse", "SciverseClient"),
    ("tools.mineru", "MinerUParser"),
    ("tools.embedding", "EmbeddingModel"),
    ("tools.reranker", "RerankerModel"),
    ("rules.unit_conversion", "UnitConverter"),
    ("rules.formula_normalizer", "FormulaNormalizer"),
    ("rules.conflict_detector", "ConflictDetector"),
    ("knowledge_base.vector_store", "VectorStore"),
    ("knowledge_base.relational", "RelationalStore"),
    ("agents.task_planning", "TaskPlanningAgent"),
    ("agents.literature_search", "LiteratureSearchAgent"),
    ("agents.literature_filter", "LiteratureFilterAgent"),
    ("agents.pdf_parsing", "PDFParsingAgent"),
    ("agents.knowledge_extraction", "KnowledgeExtractionAgent"),
    ("agents.knowledge_fusion", "KnowledgeFusionAgent"),
    ("agents.gap_identification", "GapIdentificationAgent"),
    ("agents.evidence_verification", "EvidenceVerificationAgent"),
    ("agents.report_generation", "ReportGenerationAgent"),
    ("workflow.nodes", "create_all_nodes"),
    ("workflow.engine", "MatResearcherWorkflow"),
]

for mod_name, class_name in checks:
    total += 1
    try:
        full_name = "matresearcher." + mod_name
        mod = __import__(full_name, fromlist=[class_name])
        getattr(mod, class_name)
        passed += 1
        print("OK: " + full_name)
    except Exception as e:
        failed.append((mod_name, str(e)))
        print("FAIL: " + mod_name + " -> " + str(e))

print()
print("=" * 50)
print("{}/{} passed, {} failed".format(passed, total, len(failed)))
if not failed:
    print("ALL MODULES IMPORT SUCCESSFULLY!")
elif len(failed) > 0:
    print("failures:")
    for f in failed:
        print("  - " + f[0])
sys.exit(0 if not failed else 1)
