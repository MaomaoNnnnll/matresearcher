"""Import verification test for all MatResearcher modules."""
import sys
sys.path.insert(0, 'src')

errors = []

def try_import(desc, module_path, names):
    try:
        mod = __import__(module_path, fromlist=names)
        for n in names:
            getattr(mod, n)
        print(f'  OK  {desc}')
    except Exception as e:
        errors.append(f'{desc}: {e}')
        print(f'  FAIL {desc}: {e}')

print("=== Models ===")
try_import("models.literature", "matresearcher.models.literature",
    ["Literature", "LiteratureMetadata", "ParsedDocument"])
try_import("models.knowledge", "matresearcher.models.knowledge",
    ["KnowledgeRecord", "NormalizedRecord", "FusedKnowledgeTable", "NumericValue"])
try_import("models.gap", "matresearcher.models.gap",
    ["ResearchGap", "GapScore", "ConflictItem", "MissingItem"])

print("=== Tools ===")
try_import("tools.llm", "matresearcher.tools.llm", ["LLMClient"])
try_import("tools.sciverse", "matresearcher.tools.sciverse", ["SciverseClient"])
try_import("tools.mineru", "matresearcher.tools.mineru", ["MinerUParser"])
try_import("tools.embedding", "matresearcher.tools.embedding", ["EmbeddingModel"])
try_import("tools.reranker", "matresearcher.tools.reranker", ["RerankerModel"])

print("=== Rules ===")
try_import("rules.unit_conversion", "matresearcher.rules.unit_conversion", ["UnitConverter"])
try_import("rules.formula_normalizer", "matresearcher.rules.formula_normalizer", ["FormulaNormalizer"])
try_import("rules.conflict_detector", "matresearcher.rules.conflict_detector", ["ConflictDetector"])

print("=== Knowledge Base ===")
try_import("kb.vector_store", "matresearcher.knowledge_base.vector_store", ["VectorStore"])
try_import("kb.relational", "matresearcher.knowledge_base.relational", ["RelationalStore"])

print("=== Agents ===")
try_import("agents.task_planning", "matresearcher.agents.task_planning", ["TaskPlanningAgent"])
try_import("agents.literature_search", "matresearcher.agents.literature_search", ["LiteratureSearchAgent"])
try_import("agents.literature_filter", "matresearcher.agents.literature_filter", ["LiteratureFilterAgent"])
try_import("agents.pdf_parsing", "matresearcher.agents.pdf_parsing", ["PDFParsingAgent"])
try_import("agents.knowledge_extraction", "matresearcher.agents.knowledge_extraction", ["KnowledgeExtractionAgent"])
try_import("agents.knowledge_fusion", "matresearcher.agents.knowledge_fusion", ["KnowledgeFusionAgent"])
try_import("agents.gap_identification", "matresearcher.agents.gap_identification", ["GapIdentificationAgent"])
try_import("agents.evidence_verification", "matresearcher.agents.evidence_verification", ["EvidenceVerificationAgent"])
try_import("agents.report_generation", "matresearcher.agents.report_generation", ["ReportGenerationAgent"])

print("=== Workflow ===")
try_import("workflow.engine", "matresearcher.workflow.engine", ["MatResearcherWorkflow"])
try_import("workflow.nodes", "matresearcher.workflow.nodes", ["create_all_nodes"])
try_import("state", "matresearcher.state", ["WorkflowState"])

print("=== CLI ===")
try_import("main", "matresearcher.main", ["app"])

print()
print(f"Result: {26 - len(errors)}/26 passed, {len(errors)} failed")
if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL MODULES IMPORT SUCCESSFULLY!")
