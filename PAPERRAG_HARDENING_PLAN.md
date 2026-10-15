# Save the final PaperRAG hardening plan in the project directory

## Context
The user requests the completed implementation plan as a Markdown document in the PaperRAG project directory. This is a document-only action: save the finalized plan without implementing any of its proposed code, configuration, UI, evaluation, dependency, or test changes.

## Approach
1. Read the finalized source document at `local://paper-rag-hardening-plan.md` using the file tools. This source already exists and contains the evidence analysis, selected scope, exact implementation contracts, ordered work, and verification. Copy its complete Markdown text verbatim; do not execute its Approach or Verification sections.
2. Save that text as `/home/kstv/PaperRAG/PAPERRAG_HARDENING_PLAN.md` using `write`. This destination was checked during planning and did not exist. Before writing, recheck the exact destination. If an identical file now exists, leave it unchanged. If a different file exists, preserve it and use the first unused suffix starting at `PAPERRAG_HARDENING_PLAN-2.md`, then `-3.md`, etc.
3. Change no other project/system files. Do not run application commands, installs, tests, live model requests, or the future implementation steps. Keep both session-local plan documents intact.

## Verification
Confirm the saved document is identical to the finalized source using `cmp` or equivalent exact text comparison; internal `local://` paths work with the core file tools. Confirm the document includes Context, Observed evidence, Approach, Verification, and Assumptions and contingencies, and retains continued labeled brainstorming on insufficient evidence. Report the actual saved project-relative filename. If the source cannot be read, report that exact path/error and do not invent or reconstruct the plan from memory.

## Assumptions and contingencies
The project root is `/home/kstv/PaperRAG`; only the document copy is authorized by this save action. Implementation remains proposed for a separate user instruction.
