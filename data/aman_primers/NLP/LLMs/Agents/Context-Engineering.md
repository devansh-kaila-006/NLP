# Context Engineering

**Source**: https://aman.ai/primers/ai/context-engineering
**Category**: NLP/LLMs/Agents
**Word Count**: 12001

---

Overview
Why Context Engineering
Context vs. prompt engineering
Context as a Finite Resource
Why Long Context Is Hard
The engineering objective: maximize signal, minimize noise
What context engineering is not
Mental Model
The Building Blocks of Context
System prompts
Prompt altitude: too specific, too vague, and just right
User Input as Interface
Structured outputs
Few-shot examples
Tools as context shapers
Dynamic fields
Retrieval: Turning external knowledge into on-demand context
Pre-retrieval vs. just-in-time retrieval
Memory: Retrieval with persistence across turns
State and history: Necessary for revision, reflection, and long-horizon work
The real design surface
Retrieval, Memory, Compression, and Long-Horizon Design
Retrieval: First line of defense against stale context
Curating Retrieved Context
Preloaded vs. just-in-time retrieval
Progressive disclosure
Memory as retrieval
Memory vs. State
Context compression
Compaction vs. Summarization
Tool-result clearing: Lightest safe form of compression
Structured note-taking: Externalized working memory
Sub-agents as context-management
Continuity for Long-Horizon Tasks
The core principle
Tools, Schemas, Planning Loops, and Evaluation
Tools: Mechanism to make context dynamic
Tool ambiguity
Parameter naming
Schemas for validity
Examples plus schemas
Planning as an Intermediate Artifact
Planning Improves Control
Explicit Temporal Reasoning
Why Evaluation Matters
Multi-Objective Evaluation
Failure-Driven Iteration
Planning as Context Refinement
Strong systems Are Explicit
Advanced Patterns and Failure Modes
Beyond Basic Prompting
Long-Context Failure Modes
Context poisoning
Context distraction
Context confusion
Context clash
Context compression
Memory hierarchies
Progressive disclosure for exploration
Why sub-agents help
Production Heuristics
Start minimal
Optimize token efficiency
Separate Facts from Traces
Evaluate the Whole System
The Simplest Heuristic
Implementation Playbook
Implementation Sequence
Step 1: define the task boundary before writing the prompt
Step 2: write the minimal complete system prompt
Step 3: structure the input so the model can parse the task correctly
Step 4: lock down the output contract early
Step 5: add tools only when they remove inference burden from the model
Step 6: use retrieval to keep the live context small and current
Step 7: store durable artifacts, not raw traces
Step 8: introduce compaction as soon as the agent starts carrying dead weight
Step 9: use sub-agents when they isolate context better than a single loop
Step 10: build evaluation around failure traces, not only benchmark averages
Design checklist
Final synthesis
References
Context Engineering Blogs/Guides
Prompting, in-context learning, and reasoning
Retrieval, long-context behavior, and compression
Tool use, acting, and execution
Memory, agents, and long-horizon systems
Citation
Overview
Context engineering is the discipline of deciding what information should be inside a model’s context window at the moment of inference so that the model is most likely to produce the desired behavior. In practice, that means designing, filtering, ordering, structuring, and refreshing the tokens that surround a model at run time, rather than treating the prompt as a single static instruction. This broader framing has become prominent because modern systems increasingly depend on more than a system prompt alone: they also rely on tools, retrieved documents, state, memory, schemas, and interaction history.
Effective context engineering for AI agents
describes this as optimizing the “configuration of context” for reliable agent behavior, while
Context Engineering Guide
frames it as designing and optimizing instructions plus relevant context for LLMs and other advanced AI models.
A useful way to think about it is this: prompt engineering asks “what should I say to the model?”, while context engineering asks “what should the model know, see, remember, and ignore right now?” That distinction matters because many failures in agentic systems are not caused by badly worded instructions, but by missing state, stale retrieved evidence, overlong message history, ambiguous tool definitions, or poorly structured outputs.
Effective context engineering for AI agents
emphasizes that context engineering is iterative and happens every time we decide what to pass to the model, not just once when a prompt is written.
Why Context Engineering
The term emerged because real-world LLM applications increasingly operate across multiple turns, tool calls, and time horizons. A single-shot chat prompt is often enough for simple classification or generation, but agents need a managed context state that evolves over time. That state can include system instructions, tool specifications, retrieved files, user goals, message history, planning artifacts, memory summaries, and execution traces. The recent engineering perspective is that the hard problem is no longer merely writing a clever instruction, but curating a high-signal working set from a much larger universe of potentially relevant information. That is the central theme of both
Effective context engineering for AI agents
and
The following figure (
source
) shows a high-level map of context engineering as the umbrella that contains prompting, retrieval, memory, state, and structured outputs.
Context vs. prompt engineering
Prompt engineering remains a subset of context engineering, not a competing idea. Prompt engineering focuses on writing and organizing instructions, especially system prompts. Context engineering includes that, but also covers how the prompt is combined with retrieved evidence, examples, tool signatures, state, and dynamic runtime information.
Language Models are Few-Shot Learners
by Brown et al. (2020) is a foundational reminder that model behavior is heavily shaped by in-context examples and instructions without weight updates, which is one of the clearest early demonstrations that the conditioning context itself is a powerful control surface.
A compact formal view is:
\[p(y \mid c)\]
where \(c\) is the full context presented to the model and \(y\) is the output sequence. Context engineering improves behavior by changing \(c\), not by retraining the model parameters. For autoregressive generation, the model samples token by token:
\[p(y \mid c) = \prod_{t=1}^{T} p(y_t \mid c, y_{<t})\]
so every token in the context can influence every subsequent prediction. That is why seemingly small decisions, such as whether a document snippet is included, where it is placed, or whether a tool description is ambiguous, can materially change outputs. This conditioning perspective follows directly from autoregressive language modeling and the in-context learning setup discussed in
Language Models are Few-Shot Learners
by Brown et al. (2020).
Context as a Finite Resource
A core idea in modern context engineering is that context is not a free good. More tokens can help, but beyond a point they can also dilute attention, bury relevant evidence, and introduce contradictions.
Effective context engineering for AI agents
explicitly argues that context should be treated as a finite resource with diminishing marginal returns, and
Context Rot: How Increasing Input Tokens Impacts LLM Performance
reports that model performance can degrade as input length grows, even on simple tasks.
This is consistent with long-context evaluation work.
Lost in the Middle: How Language Models Use Long Contexts
by Liu et al. (2024) shows that many models are sensitive not just to whether relevant information is present, but also to where it appears, with strong performance often at the beginning or end of context and weaker performance when crucial evidence sits in the middle. In other words, context quality depends on placement and salience, not just inclusion.
The following figure (
source
) shows the shift from single-turn prompt engineering to iterative context curation for agents that combine prompts, tools, memory, documents, and message history.
Why Long Context Is Hard
The architectural reason long context is expensive is rooted in the Transformer. In standard self-attention, each token can attend to every other token, so the number of pairwise interactions scales quadratically with sequence length.
Attention Is All You Need
by Vaswani et al. (2017) introduced the Transformer and the scaled dot-product attention mechanism:
\[\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\]
A useful engineering approximation is that the raw interaction burden grows like:
\[O(n^2)\]
for \(n\) context tokens. That does not mean every long-context model fails immediately, but it does explain why attention gets increasingly precious as context grows and why the model may allocate it imperfectly. The Anthropic engineering note connects this to an “attention budget” intuition, while the Chroma “context rot” report and Liu et al.’s long-context evaluation both show practical degradation patterns as contexts become larger or less well structured.
The engineering objective: maximize signal, minimize noise
The most useful operational definition is: context engineering tries to find the smallest high-signal set of tokens that maximizes the probability of the desired outcome. That principle appears almost verbatim in
Effective context engineering for AI agents
, and it aligns with the broader practitioner view in
Context Engineering Guide
, which emphasizes optimizing what enters the context window and filtering noisy information through experimentation and evaluation.
This objective has two immediate consequences:
First, adding information is not always helpful. A document that is technically relevant but weakly aligned to the current subtask may reduce performance more than it helps. That is one reason retrieval systems increasingly combine ranking, chunking, summarization, and schema constraints rather than dumping raw corpora into context.
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
by Lewis et al. (2020) is a canonical reference here, showing that non-parametric memory can improve factual generation by selectively bringing external evidence into the model’s conditioning context.
Second, the order and structure of information matter. Clear delimiters, tagged sections, JSON schemas, and tool descriptions can reduce ambiguity by making the latent task structure easier for the model to infer. This is a recurring theme in both recent practitioner guides, which treat structured inputs, output schemas, tool contracts, and dynamic fields such as time or user state as first-class parts of the context design problem.
What context engineering is not
Context engineering is not a new training objective, and it is not a replacement for model quality, evals, or product design. Most base language models are still trained with next-token prediction, usually via cross-entropy over the target token sequence. A generic autoregressive training loss is:
\[\mathcal{L}_{\text{NLL}} = - \sum_{t=1}^{T} \log p(y_t \mid c, y_{<t})\]
At deployment time, context engineering changes \(c\), while fine-tuning or continued pretraining changes the model parameters themselves. That distinction is important because many teams overestimate what prompting alone can fix and underestimate what better retrieval, shorter history, cleaner tools, or stronger evaluation can fix. The practitioner literature around context engineering repeatedly frames it as an inference-time systems problem layered on top of model capability.
Language Models are Few-Shot Learners
by Brown et al. (2020) is a useful anchor for this distinction because it highlights how much can change from context alone, while current engineering guidance makes clear that this control still has limits.
Mental Model
A practical mental model is to treat the model like a temporary reasoning process with a limited, expensive, and position-sensitive working memory. Your job is to assemble the right workspace for the next step. That workspace usually has five questions behind it:
What must the model know right now?
What should it ignore right now?
What can be loaded only on demand?
What structure will make the task legible?
What evidence will let you tell whether the chosen context actually improved performance?
That last question matters because context engineering is inherently empirical. A beautifully designed context is only good if it improves task success, reliability, latency, cost, or safety under evaluation. Recent practitioner writing consistently treats evals as inseparable from context design, because stale, bloated, or ambiguous context often fails in subtle ways that only show up under systematic testing.
Context Engineering Guide
stresses evaluation as the way to know whether context tactics are working, and Anthropic’s engineering guidance frames the same challenge as selecting the smallest high-signal context that reliably produces the desired behavior.
The Building Blocks of Context
System prompts
The system prompt is the highest-leverage static component in most context stacks because it defines the model’s role, task boundary, response contract, and tool-use posture before any user-specific information arrives. In agent settings, the best system prompts are neither brittle pseudo-programs nor vague mission statements. They sit in a middle zone: explicit enough to constrain behavior, but abstract enough to generalize across cases. Recent engineering guidance recommends clear sectioning, direct language, and the smallest complete specification of expected behavior, while the planning example in the source material shows exactly this pattern: role declaration, task framing, required fields, and formatting expectations are all made explicit up front.
A useful abstraction is that the system prompt specifies a policy over outputs and actions conditioned on the current context \(c\). If \(a\) denotes the next action, such as answering, asking for clarification, or calling a tool, then prompt quality affects the induced policy
\[a^* = \arg\max_a p(a \mid c)\]
That framing is closely aligned with the original in-context learning perspective of
Language Models are Few-Shot Learners
by Brown et al. (2020), which showed that behavior can be strongly steered by instructions and demonstrations without changing model weights, and with Anthropic’s later view that the broader problem is selecting the full token configuration that induces the desired behavior.
Prompt altitude: too specific, too vague, and just right
One of the most practical ideas in modern context engineering is prompt altitude. If a prompt hardcodes too much workflow logic, it becomes fragile, difficult to maintain, and prone to failure when reality deviates from the examples. If it is too generic, the model lacks enough signal to infer the intended structure of the task. Anthropic explicitly describes the optimal prompt as the “Goldilocks zone” between brittle if-else prompting and underspecified guidance, while the planning example in the source material shows how this balance is achieved by combining a concise mission statement with exact field requirements and output constraints.
In practice, “just right” usually means decomposing the system prompt into legible sections such as
<background_information>
,
<instructions>
,
## Tool guidance
,
# Output description
, etc. That recommendation matches the planning example’s structure, where the model is told what role it has, what the input delimiter is, what fields each subtask must contain, and how to infer derived fields such as dates.
The following figure (
source
) shows the calibration problem for system prompts: overly specific prompts become brittle, while overly vague prompts leave too much latent ambiguity.
User Input as Interface
A strong primer-level insight is that user input should not be treated as an unstructured blob. It is better understood as one typed component inside a larger protocol. In the planning example, the user request is wrapped in explicit delimiters, which separates task data from instructions and makes the mapping from input query to output subtasks easier for the model to infer. The source material explicitly notes that delimiters reduce confusion and clarify what should be transformed.
This idea is closely related to the original few-shot prompting paradigm in
Language Models are Few-Shot Learners
by Brown et al. (2020), where task instructions, examples, and query are arranged in a format that implicitly teaches the model the task. The lesson for context engineering is that input formatting is itself supervision. XML tags, markdown headers, and typed fields all help compress ambiguity into structure the model can exploit.
Structured outputs
Once a model response has to feed another component, free-form prose becomes a liability. The planning example therefore specifies required output fields such as (id), (query), (source_type), (time_period), (domain_focus), and (priority), and then adds a JSON example so that an output parser can derive a schema and validate the result. The explicit motivation is consistency: if the next workflow node expects structured fields, context should teach the model that contract before generation begins.
At a systems level, schema-constrained generation reduces the entropy of the output space. If \(y\) is the raw response and (\mathcal{S}) is the allowed schema, then structured output is effectively trying to maximize
\[p(y \in \mathcal{S} \mid c)\]
rather than merely maximizing general fluency. This is why structured outputs are especially valuable in pipelines, tool orchestration, and evaluation harnesses. The planning example even illustrates how slight ambiguity can leak through, as shown by a generated priority value of 1.2 despite an intended integer scale, which is a good reminder that schemas should be enforced both by prompt design and by downstream validation.
Few-shot examples
Examples remain one of the most reliable ways to steer behavior, but the quality of examples matters more than the sheer count. Anthropic’s guidance is explicit here: do not stuff every edge case into the prompt. Instead, use a compact set of diverse, canonical examples that show the desired behavior clearly. The source material makes the same point indirectly through the planning prompt, where compact examples of field formats and output shape do much of the steering work.
This principle is grounded in classic in-context learning results.
Language Models are Few-Shot Learners
by Brown et al. (2020) established that demonstrations can act as task conditioning without gradient updates, and
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022) showed that carefully chosen reasoning-plus-action exemplars can teach a model how to alternate between thought and external action in multi-step tasks. In context-engineering terms, examples are compact behavioral evidence, and good examples often outperform long prose rules because they specify both format and policy in one shot.
Tools as context shapers
Tools do more than expand what an agent can do. They determine what new information can enter the context window, when it enters, and in what representation. Anthropic’s engineering guidance emphasizes that tools define the contract between the agent and its action or information space, so they should be self-contained, robust, token-efficient, and minimally overlapping. The source material gives a simple but instructive example: a date-time capability is treated as a tool-like dynamic context source because the model otherwise guesses temporal boundaries poorly.
The research literature supports this shift toward tool-mediated reasoning.
Toolformer: Language Models Can Teach Themselves to Use Tools
by Schick et al. (2023) presents a model trained to decide when to call APIs and how to integrate results into subsequent prediction, while
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022) shows that interleaving reasoning with actions can improve both accuracy and interpretability on knowledge and decision tasks. In practical context engineering, that means tool descriptions, parameter names, return payloads, and invocation heuristics are all part of the prompt surface.
Dynamic fields
A recurring beginner mistake is to assume the model will infer temporal or environmental facts that are not explicitly present. The planning example corrects this by injecting the current timestamp and requiring derived (start_date) and (end_date) fields from coarse time labels such as “recent” or “last week.” The source material is blunt about why: without current date information, the model tends to guess, leading to weaker searches and weaker downstream performance.
This is a broader design rule: any variable that changes over time and materially affects the task should either be injected explicitly or made available through a tool. Date, locale, user permissions, active file path, model settings, and workflow stage are all examples of dynamic context. Context engineering turns these from hidden assumptions into explicit state.
Retrieval: Turning external knowledge into on-demand context
Retrieval is one of the most important bridges between fixed model weights and changing world knowledge. The source material explicitly includes retrieval-augmented generation and vector-store lookup as core context-engineering techniques, and the planning example discusses storing previously generated subqueries in a vector store so similar future queries can reuse them instead of paying for redundant LLM calls.
The canonical research reference is
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
by Lewis et al. (2020), which combines a parametric generator with non-parametric memory so the model can condition on retrieved passages rather than relying only on weights. At inference time, the retrieval step can be summarized as selecting documents
\[d^* = \arg\max_{d \in \mathcal{D}} \mathrm{sim}(q,d)\]
where \(q\) is the current query, \(\mathcal{D}\) is the document store, and \(\mathrm{sim}\) is a relevance function such as dense-vector similarity. Context engineering begins after retrieval, because the retrieved items still need chunking, ranking, trimming, ordering, and formatting before they become useful context.
Pre-retrieval vs. just-in-time retrieval
A major design choice is whether to load candidate context before the agent begins reasoning, or let the agent retrieve incrementally while it works. Anthropic’s engineering writeup describes a growing shift toward “just in time” context strategies, where the agent keeps lightweight references such as file paths, URLs, or stored queries and loads detailed content only when needed. That pattern is presented as more storage-efficient and closer to how humans rely on external organization rather than memorizing everything in working memory.
The engineering trade-off is straightforward: pre-retrieval improves latency and predictability, while just-in-time retrieval improves adaptability and token efficiency. Hybrid designs often win in practice, especially when there is a small amount of universally relevant context and a much larger body of situational context. Anthropic explicitly recommends hybrid retrieval in some settings, and the planning example’s vector-store reuse fits neatly into that logic.
Memory: Retrieval with persistence across turns
Memory is often discussed vaguely, but in context-engineering terms it is best understood as persisted information that can be selectively reintroduced into context later. The source material distinguishes short-term memory from long-term memory and gives a practical example of cached subqueries stored in a vector store for reuse, mainly to reduce latency and cost. That is a concrete memory design: store artifacts that are expensive to regenerate and likely to recur.
In a more formal sense, memory selection can be described as choosing a subset \(M_t\) of stored items \(M\) relevant to the current turn \(t\):
\[M_t = \operatorname{TopK}_{m \in M} , s(m, c_t)\]
where \(s\) is a relevance score against the current context \(c_t\). The challenge is not merely storing more memory, but retrieving the right memory and excluding the rest. This is why context engineering treats memory as a curation problem, not a logging problem.
State and history: Necessary for revision, reflection, and long-horizon work
State is not the same thing as memory. Memory often refers to reusable facts or artifacts across sessions or turns, while state refers to the current execution trace of the task: subtasks, revisions, intermediate results, pending tool outputs, and workflow phase. The source material highlights that revision-oriented agent systems need access to previous states and historical context because they may take multiple passes at a problem and must compare current plans against prior attempts.
This requirement becomes acute in long-horizon settings, where the agent cannot carry the full history verbatim forever. Anthropic’s engineering guidance frames long-horizon work as a context-window management problem and argues that context must stay informative but tight. The implication is that raw message history should eventually be compressed into summaries, checkpoints, or structured state records that preserve task-relevant commitments without preserving every token.
The following figure (
source
) shows a multi-stage agent workflow in which planning, orchestration, summarization, and delivery each create state that may need to be passed forward or revisited later.
The real design surface
The most important practical lesson is that these components do not work independently. The system prompt tells the model how to interpret the user input; delimiters make the input legible; examples show the latent mapping; tools decide what additional information can be brought in; retrieval chooses external evidence; memory decides what past artifacts are worth resurfacing; and structured outputs ensure that the result can be consumed by the next step. Both recent engineering writeups converge on the same principle: successful systems come from curating a compact, high-signal, dynamically refreshed context stack rather than from optimizing any single line of prompt text in isolation.
Retrieval, Memory, Compression, and Long-Horizon Design
Retrieval: First line of defense against stale context
For most nontrivial agent systems, retrieval is the mechanism that keeps context both current and selective. Instead of forcing the model to rely entirely on parametric memory, retrieval lets the system surface external evidence only when it is relevant to the current subtask. This is why modern context engineering treats retrieval not as an optional add-on, but as one of the main controls for trading off freshness, precision, latency, and cost. The core practical idea is simple: keep the live context window small, but make it easy to pull in the right evidence when needed, a view emphasized in both
Effective context engineering for AI agents
and
The canonical technical reference is
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
by Lewis et al. (2020), which shows that combining a generator with retrieved external passages improves performance on knowledge-intensive tasks by turning external documents into part of the conditioning context rather than expecting the model to memorize everything in its weights.
A standard retrieval formulation is to rank documents \(d \in \mathcal{D}\) against a query \(q\) using a relevance score \(s(q,d)\), then pass the top \(k\) results into the context:
\[\mathcal{D}_k = \operatorname{TopK}_{d \in \mathcal{D}} , s(q,d)\]
In dense retrieval systems, \(s(q,d)\) is often a vector similarity such as cosine similarity:
\[s(q,d) = \frac{e(q)^\top e(d)}{|e(q)| |e(d)|}\]
where \(e(\cdot)\) is the embedding function. The context-engineering problem begins after this ranking step, because the system still has to decide how much of each document to include, in what order, and with what surrounding instructions.
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
by Lewis et al. (2020) is especially relevant here because it makes clear that retrieval quality and generation quality are tightly coupled.
Curating Retrieved Context
A common failure mode is to retrieve correctly and still lose performance because the retrieved material is too long, too redundant, or too weakly aligned to the immediate decision. Context engineering therefore treats retrieval as a multistage process: fetch, filter, compress, order, and frame. Recent engineering guidance explicitly warns that context can become stale, irrelevant, or diluted, which means retrieval must be paired with evaluation and pruning rather than assumed to be helpful by default.
Context Engineering Guide
highlights exactly this issue when it notes that context can become inefficient and requires evaluation workflows to catch it, while
Effective context engineering for AI agents
argues that the target is the smallest high-signal set of tokens that still supports the desired behavior.
This is closely related to long-context evidence from
Lost in the Middle: How Language Models Use Long Contexts
by Liu et al. (2024), which shows that model performance is sensitive not only to whether relevant evidence is present, but also to where it is placed in the context window. In practice, that means retrieved passages should usually be ranked not just by semantic relevance, but by expected decision value for the current step.
Preloaded vs. just-in-time retrieval
One of the most important design choices is whether to retrieve up front or retrieve during execution. A preloaded strategy fetches likely-relevant material before reasoning starts, which improves latency and predictability but risks overloading the context with material that turns out not to matter. A just-in-time strategy keeps lightweight references, such as file paths, URLs, or stored queries, and loads content only when the agent decides it needs it. Recent engineering guidance strongly emphasizes this latter pattern for agents because it reduces unnecessary token load and supports exploration over large information spaces.
Effective context engineering for AI agents
describes exactly this shift toward lightweight identifiers plus runtime loading, and
Context Engineering Guide
presents vector-store reuse and selective retrieval as cost-saving, latency-saving examples of good context design.
The most robust production pattern is often hybrid retrieval: preload a small amount of universally relevant context, then let the agent retrieve further detail on demand. That hybrid approach is explicitly recommended in recent practitioner guidance because some tasks benefit from immediate grounding, while others require progressive exploration.
Effective context engineering for AI agents
is especially clear that the “right” degree of autonomy depends on the task and that the simplest working design is often best.
Progressive disclosure
Just-in-time retrieval becomes most powerful when paired with progressive disclosure. Instead of showing the agent all potentially relevant information at once, the system lets the agent expose detail layer by layer. Metadata, filenames, timestamps, chunk titles, or section headers often supply enough signal for the next step without requiring the full underlying document to enter the context. Recent engineering guidance explicitly describes this as a way to let agents assemble understanding incrementally and keep only what is necessary in working memory.
Effective context engineering for AI agents
uses file hierarchies, naming conventions, and timestamps as examples of this behavior-guiding metadata.
This pattern is valuable because it aligns with the finite-context view. If the agent sees only the highest-yield evidence at each step, it is less likely to drown in exhaustive but low-value material. That is one reason progressive disclosure is increasingly central to agentic search and coding systems.
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022) is relevant here because it shows that alternating internal reasoning with environment actions can outperform pure reasoning or pure acting baselines, especially when the external action retrieves new information that sharpens the next reasoning step.
Memory as retrieval
In context engineering, memory is most useful when it stores distilled artifacts that are likely to matter later, not when it stores every raw token from the past. A good memory system therefore looks more like indexed retrieval over summaries, notes, plans, decisions, and durable facts than like an ever-growing chat log. This practical view appears clearly in current engineering guidance, which distinguishes between transient working context and persisted information that can be consulted later, and in the planning example where cached subqueries in a vector store are reused to avoid repeated LLM calls.
Context Engineering Guide
highlights this subquery-caching pattern, while
Effective context engineering for AI agents
discusses note-taking and external memory for long-horizon coherence.
A simple memory retrieval rule is
\[M_t = \operatorname{TopK}_{m \in M} , s(m, c_t)\]
where \(M\) is the persistent memory store, \(c_t\) is the current context at turn \(t\), and \(s\) scores the relevance of memory item \(m\) to the current step. The critical design question is not “how much can we store?” but “what should be written, how should it be represented, and when should it be recalled?”
A strong systems reference here is
MemGPT: Towards LLMs as Operating Systems
by Packer et al. (2023), which frames limited context windows as a memory hierarchy problem and proposes virtual context management, where the model works with multiple memory tiers rather than pretending that everything important can remain in a single active window. That paper is especially relevant because it moves the discussion from ad hoc summarization toward explicit memory management.
Memory vs. State
Short-term memory refers to information that must remain available over the immediate next few turns, such as active subtasks, open hypotheses, recent tool outputs, and unresolved ambiguities. Long-term memory refers to information that persists across longer time spans, such as user preferences, project conventions, reusable plans, or previously successful subtasks. State refers to the agent’s current execution trace: what stage it is in, what has already been tried, what is pending, and what changed after the latest tool call. The distinction matters because each type of information deserves a different retention and compression policy. Recent practitioner guidance makes this distinction explicit by separately discussing state or historical context, vector-store memory, note-taking, and compaction for long interactions.
Effective context engineering for AI agents
both treat these as separate design surfaces.
An especially important point from the planning-oriented view is that revision loops require access to prior states, not just final answers. If an agent may revise subtasks, rerun searches, or revisit earlier hypotheses, then previous versions of the plan and their outcomes become part of the useful context state.
Context Engineering Guide
stresses exactly this in its discussion of revisions and historical context for report generation workflows.
Context compression
Once a task stretches across many turns, raw history becomes too large and too noisy to preserve verbatim. At that point the system needs compaction. The goal of compaction is not merely to shorten context, but to preserve decision-relevant content while discarding low-value tokens such as stale tool outputs, redundant observations, or resolved intermediate details. Recent engineering guidance treats compaction as one of the main techniques for long-horizon tasks and recommends starting with high recall, then improving precision by pruning superfluous content.
Effective context engineering for AI agents
is particularly concrete on this point.
A general compaction objective can be written as selecting a compressed summary \(z\) of history \(h\) that maximizes retained utility under a token budget \(B\):
\[z^* = \arg\max_{z : |z| \le B} U(z; h, \tau)\]
where \(U\) measures usefulness for future task \(\tau\). In practice, \(U\) is not directly observable, so compaction systems are usually optimized through evaluation on real traces rather than through a closed-form objective.
A directly relevant compression paper is
LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
by Jiang et al. (2023), which proposes coarse-to-fine prompt compression and shows that prompt length can be reduced substantially while retaining much of the useful content. The paper is valuable here because it treats compression as an inference-time optimization problem rather than as a purely linguistic summarization problem.
Compaction vs. Summarization
Summarization aims to produce a readable synopsis for humans or models. Compaction aims to preserve future task utility under severe token constraints. Those objectives overlap, but they are not identical. A beautiful summary can omit the one detail that later becomes decisive, while an ugly but high-recall compaction can still be the right engineering choice. This is why recent guidance emphasizes tuning compaction prompts on complex traces and being cautious about aggressive pruning.
Effective context engineering for AI agents
explicitly warns that overly aggressive compaction can discard subtle context whose importance only becomes clear later.
A practical decomposition is to preserve four classes of content during compaction: active goals, unresolved uncertainties, durable decisions, and references to where deeper detail can be reloaded. That last category is especially important because it lets the system drop raw detail while preserving the path to recover it. In other words, good compaction is often paired with good retrieval.
Tool-result clearing: Lightest safe form of compression
One of the safest compaction operations is to clear or heavily compress raw tool results once their informational content has been absorbed into a state summary or note. Clearing tool results is a lightweight and safe compaction strategy, making it low-hanging fruit. Once a tool result sits deep in the history, the agent usually does not need the entire raw payload repeated verbatim.
The logic is straightforward. If a tool call output \(r\) has already been transformed into a compact state update \(u(r)\), then future context should usually retain \(u(r)\) rather than \(r\) itself, provided that the system can re-fetch \(r\) if necessary. This reinforces the broader principle that active context should carry conclusions and references, not raw exhaust.
Structured note-taking: Externalized working memory
Structured note-taking is a particularly effective for maintaining coherence across long-horizon tasks because it turns ephemeral reasoning into persisted, queryable artifacts. Instead of keeping all progress inside the model’s transient context, the agent writes notes, plans, to-do lists, checkpoints, or decision logs to an external memory store and later reads back only the relevant portions. Recent engineering guidance strongly endorses this pattern and gives examples in which persistent notes allow the agent to resume complex multi-hour behavior after context resets.
This design is powerful because it makes memory explicit and inspectable. It also gives the system a natural place to store task abstractions rather than only raw observations.
MemGPT: Towards LLMs as Operating Systems
by Packer et al. (2023) is relevant again here because it provides a broader memory-hierarchy perspective, while recent practitioner guidance shows how simple file-based notes can already deliver much of the benefit in practice.
Sub-agents as context-management
Sub-agents are often introduced as a way to parallelize work, but they are just as important as a way to isolate context. A specialized sub-agent can explore deeply inside its own narrow context window, then return a compact summary to a coordinating agent. This keeps the main agent’s working context cleaner and prevents global overload. Recent engineering guidance explicitly describes sub-agent architectures as a way around context limitations, because the lead agent no longer has to carry every low-level detail from every branch of exploration.
This is closely aligned with the behavior observed in
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022), where the system alternates between internal reasoning and external actions, and with tool-use frameworks such as
Toolformer: Language Models Can Teach Themselves to Use Tools
by Schick et al. (2023), which show that models can learn to invoke tools and integrate their outputs into later prediction. In context-engineering terms, sub-agents are a disciplined way to localize exploration and return only distilled evidence to the top-level policy.
Continuity for Long-Horizon Tasks
Long-horizon tasks fail when the system cannot preserve enough continuity across many steps. That continuity can come from compaction, note-taking, memory retrieval, sub-agents, or checkpointed state. Recent engineering guidance is explicit that larger context windows alone do not eliminate this problem. Even when models can technically ingest very long sequences, effective agents still need mechanisms that preserve coherence, salience, and task direction over time.
Effective context engineering for AI agents
,
Lost in the Middle: How Language Models Use Long Contexts
by Liu et al. (2024), and
LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
by Jiang et al. (2023) all point, from different angles, to the same reality: longer raw context does not automatically produce better effective context.
A useful engineering summary is that long-horizon capability depends on three loops working together:
\[\text{observe} \rightarrow \text{compress} \rightarrow \text{retrieve}\]
with occasional branching into:
\[\text{delegate} \rightarrow \text{summarize} \rightarrow \text{synthesize}\]
That is the real architecture behind sustained agent performance.
The core principle
Whether the system uses vector retrieval, cached subqueries, external notes, compaction prompts, or sub-agents, the underlying principle is the same: preserve the minimum amount of information that is maximally useful for the next decision. Recent engineering guidance puts this principle at the center of the field, and the broader practitioner view reaches the same conclusion from a workflow and cost perspective.
Tools, Schemas, Planning Loops, and Evaluation
Tools: Mechanism to make context dynamic
In an agentic system, tools are not peripheral utilities. They are the machinery that lets the model acquire fresh context, act on the environment, and transform latent intentions into grounded steps. This is why modern context engineering treats tool design as part of the context problem itself: a tool’s name, description, arguments, return format, and usage guidance all shape what the model can infer about when and how to use it.
Effective context engineering for AI agents
emphasizes that tools define the contract between the agent and its action or information space, while
Writing effective tools for AI agents
stresses clarity, minimal overlap, and descriptive parameters.
A useful formalization is to treat a tool \(T_i\) as a function mapping structured input \(x_i\) to result \(r_i\):
\[r_i = T_i(x_i)\]
The model’s job is then partly to select
\[i^* = \arg\max_i p(T_i \mid c)\]
where \(c\) is the current context. Good tool design raises the probability of correct tool choice by making intended usage legible in-context.
Tool ambiguity
When multiple tools partially overlap, the model is forced into a fuzzy classification problem it may not have enough signal to solve. This leads to wasted context, unnecessary calls, dead ends, or contradictory execution paths. Recent engineering guidance is explicit that if a human engineer cannot state clearly which tool should be used in a given situation, an agent is unlikely to do better. That is why effective tool sets are usually small, composable, and differentiated by purpose rather than by minor implementation details.
Writing effective tools for AI agents
and
Effective context engineering for AI agents
both make this point directly.
This connects naturally to
Toolformer: Language Models Can Teach Themselves to Use Tools
by Schick et al. (2023), which demonstrates that language models can learn when to insert tool calls, but only when the tool interface is sufficiently coherent for the model to exploit. In practice, context engineering improves tool use not just by adding descriptions, but by eliminating unnecessary choice ambiguity.
Parameter naming
A common mistake is to think of tool arguments as purely software-level details. In reality, argument names and allowed values are part of the model-facing language interface. A parameter like
time_period
is easier for a model to use than a vague alternative such as
window_hint
, and an enum like \({\text{today}, \text{last week}, \text{recent}, \text{past year}, \text{all time}}\) provides a much clearer policy surface than free text. The planning example in the source material shows exactly this pattern: the model is given a field list, examples of acceptable values, and guidance on which fields may be null, all of which reduce output ambiguity and downstream repair work.
In effect, the model is solving a conditional structured prediction problem. Cleaner parameter names reduce uncertainty in the internal mapping from instruction to action, which is why tool interface design and schema design are deeply intertwined.
Schemas for validity
Schemas are among the most important stabilizers in context engineering because they transform a broad generation problem into a constrained generation problem. Instead of asking the model to “return something useful,” a schema tells it what keys must exist, what types those keys should have, which fields are optional, and often what value ranges are acceptable. The planning example shows this clearly: the system prompt specifies exact subtask fields, then a JSON example is used so that the downstream parser can infer a schema and validate the output.
Mathematically, schema-constrained generation is trying to maximize the probability mass assigned to the valid set \(\mathcal{S}\):
\[\max p(y \in \mathcal{S} \mid c)\]
This can also be seen as reducing entropy over admissible outputs. The model still predicts tokens autoregressively, but the engineering target is no longer generic fluency. It is validity, consistency, and compatibility with downstream systems.
Examples plus schemas
Schemas are good at specifying structure, but they do not always teach style or latent task policy. Examples are good at showing behavior, but they do not by themselves guarantee type consistency. The strongest systems therefore pair them. This is exactly what the planning example does: it provides a typed field specification and a concrete JSON example. Anthropic’s guidance on examples also supports this pairing by arguing for a compact set of canonical demonstrations rather than a long prose rule list.
This principle fits well with
Language Models are Few-Shot Learners
by Brown et al. (2020), where demonstrations condition behavior, and with
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022), where exemplar trajectories teach the model how to sequence thought and action. In a context-engineering workflow, schemas define the guardrails and examples define the driving style.
Planning as an Intermediate Artifact
One of the strongest recurring themes in context engineering is that models perform better when latent reasoning structure is externalized into explicit artifacts. The planning sub-agent example is a clear case: instead of going directly from a user request to a final answer, the system first produces a search plan composed of structured subtasks, each with source type, domain focus, priority, and temporal bounds. That plan then becomes context for later retrieval and synthesis steps.
This approach is closely aligned with
Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models
by Wang et al. (2023), which shows that forcing the model to first create a plan and then execute it can improve multi-step reasoning. It also complements
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022), where external actions interleave with reasoning. The context-engineering angle is that a plan is not just reasoning, it is reusable structured context for the next stage.
Planning Improves Control
Once the system represents a plan explicitly, the rest of the pipeline becomes easier to control. A coordinator can reorder subtasks, deduplicate them, drop low-priority branches, inject extra retrieval constraints, or rerun only the failed branches rather than the full workflow. This matters because many agent failures are local, not global. By separating planning from execution, context engineering makes those failures inspectable. The planning example is built exactly this way: subtasks are generated first, stored in structured form, and then reused by later workflow components.
A compact representation is
\[P = {s_1, s_2, \dots, s_n}\]
where each \(s_i\) is a structured subtask. Execution then becomes a conditional process over the plan:
\[r_i = E(s_i, c_i)\]
where \(E\) is the execution policy and \(c_i\) is the local context assembled for that subtask.
Explicit Temporal Reasoning
A subtle but very important lesson from the planning example is that time-sensitive reasoning should not be left implicit. The source material injects the current date and time into the context and then asks the planning component to derive concrete (start_date) and (end_date) values from coarse labels such as “recent” or “last week.” This is an excellent example of context engineering because it converts a vague temporal concept into an executable constraint that downstream search tools can use reliably.
This reflects a broader principle: if a latent variable matters to downstream action, make it explicit early. Time, jurisdiction, user role, platform, language, or safety mode often belong in this category. Explicit state almost always beats inferred state when correctness matters.
Why Evaluation Matters
Because context engineering is an inference-time systems discipline, almost every major tactic in it has to be validated empirically. A prompt can look clearer and still degrade outcomes. Retrieval can look more comprehensive and still bury the crucial fact. Compression can lower token count and still remove the one detail that future turns depend on. This is why recent practitioner material consistently treats evals as part of context engineering rather than as a separate concern.
Context Engineering Guide
explicitly recommends formal evaluation pipelines to measure whether tactics are working, and
Effective context engineering for AI agents
frames context selection as a performance optimization problem under hard constraints.
A generic evaluation objective is to compare two context strategies \(c_A\) and \(c_B\) over a task set \(\mathcal{T}\):
\[\Delta = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \left[m(f(c_B,\tau)) - m(f(c_A,\tau))\right]\]
where \(f\) is the system and \(m\) is a task metric such as accuracy, pass rate, latency-adjusted utility, or cost-normalized success. Context engineering is only successful when \(\Delta > 0\) for the metrics that matter.
Multi-Objective Evaluation
A context strategy that improves answer quality but doubles cost or latency may still be unacceptable in production. A retrieval strategy that increases recall but lowers precision may help research workflows and hurt customer-support workflows. For this reason, context-engineering evals are usually multi-objective. They often track task success, structural validity, tool efficiency, token cost, latency, and failure rate together. This systems view is implicit throughout recent engineering guidance, which repeatedly emphasizes trade-offs among signal, token efficiency, autonomy, and maintenance complexity.
One useful scalarized score is:
\[J = \alpha \cdot \text{quality} - \beta \cdot \text{latency} - \gamma \cdot \text{cost} - \delta \cdot \text{failure_rate}\]
where the coefficients reflect product priorities. In practice, teams often compare strategies on a Pareto frontier rather than forcing everything into one number.
Failure-Driven Iteration
The most productive way to improve context engineering is usually not to add more prompt text preemptively, but to study failure traces and patch the specific context defect that caused them. If the agent uses the wrong tool, improve tool differentiation. If it omits a required field, tighten the schema. If it misses time-sensitive content, inject explicit temporal state. If it overuses history, add compaction or summary checkpoints. This iterative workflow is exactly what current engineering guidance recommends: start with a minimal working setup, observe where it breaks, then add the smallest corrective context that addresses the observed failure mode.
This is also consistent with the evolution from prompt engineering to context engineering. The real work is not finding a magical static prompt, but tuning a dynamic context system against observed behavior.
Planning as Context Refinement
The deeper perspective is that agent loops are context loops. A model receives context, produces an action, obtains new information, and then the system has to decide what part of that new information should become part of the next context. Planning, tool use, retrieval, and evaluation are all subcases of this same process.
Effective context engineering for AI agents
describes context engineering as iterative curation at each step, and the planning example shows a concrete version of this idea where search subtasks are generated first so that later stages can operate with a cleaner and more directed context.
A minimal loop looks like:
\[c_t \rightarrow a_t \rightarrow o_t \rightarrow c_{t+1}\]
where \(c_t\) is the current context, \(a_t\) is the chosen action, and \(o_t\) is the resulting observation. Context engineering is the design of the transition rule from \((c_t, a_t, o_t)\) to \(c_{t+1}\).
Strong systems Are Explicit
The most reliable pattern across recent work is that strong context-engineered systems tend to be explicit about things that weak systems leave implicit: they specify roles, delimit inputs, constrain outputs, define tools clearly, externalize plans, surface time, track state, and evaluate changes systematically. None of those tactics is individually glamorous, but together they make agent behavior more legible, more controllable, and more robust. That is the shared practical lesson of recent engineering guidance and the planning-oriented workflow example.
Advanced Patterns and Failure Modes
Beyond Basic Prompting
Once a system already has a decent system prompt, clear delimiters, schemas, and retrieval, the next layer of work is about keeping context useful over time. The source material explicitly points to this next layer as context compression, context management, context safety, and evaluating context effectiveness over time, with the warning that context can dilute and fill with stale or irrelevant information if it is not actively managed.
At this stage, the governing problem is no longer “how do I phrase the task?” but “how do I prevent the task representation from degrading as the interaction grows?” A useful abstraction is to view context quality as a function of both relevance and freshness under a token budget \(B\):
\[Q(c) = \sum_{i=1}^{|c|} \mathrm{signal}(t_i) - \lambda \sum_{i=1}^{|c|} \mathrm{noise}(t_i), \qquad |c| \le B\]
Advanced context engineering tries to maximize \(Q(c)\) turn after turn, not just once at initialization.
Long-Context Failure Modes
A particularly useful synthesis in
How Long Contexts Fail and How to Fix Them
is the four-way breakdown of long-context failure: context poisoning, context distraction, context confusion, and context clash. These are presented as concrete ways overloaded contexts derail agentic behavior, and they provide a practical taxonomy for debugging production systems.
These failure types also fit the broader empirical picture in
Context Rot: How Increasing Input Tokens Impacts LLM Performance
by Hong et al. (2025), which reports that LLM performance changes materially as input length grows rather than remaining uniform across positions and lengths.
Context poisoning
Context poisoning happens when an early hallucination, mistaken assumption, or irrelevant plan becomes part of the running state and then keeps influencing future steps. In an agent loop, poisoned state can become self-reinforcing because later actions are conditioned on earlier outputs that were never corrected. The source material explicitly describes this as a failure mode where hallucinated or irrelevant goals persist and continue to mislead the system.
The mitigation is to insert verification and reset points. Intermediate plans, derived facts, and tool outputs should be summarized into validated state rather than copied forward blindly. This is one reason schemas, tool-result clearing, and explicit state checkpoints matter so much in long-horizon agents. Anthropic’s guidance around compaction and note-taking supports exactly this design instinct.
Context distraction
Context distraction occurs when growing history pulls the model toward what it has already done instead of what it should do next. The source material describes this as a tendency for agents in very large contexts to overfit to past actions rather than generate new strategies.
This is closely related to
Lost in the Middle: How Language Models Use Long Contexts
by Liu et al. (2024), which shows that long-context performance depends heavily on information placement, i.e., where relevant information appears in the context, and to
Context Rot: How Increasing Input Tokens Impacts LLM Performance
by Hong et al. (2025), which documents that performance does not remain stable, i.e., varies significantly, as input length grows.
The main fix is compaction with intent. Instead of carrying the full trace, compress prior work into active goals, unresolved questions, durable decisions, and reloadable references. That converts distracting history into usable working state.
Context confusion
Context confusion happens when the model receives too many tool definitions, policy fragments, or partially overlapping instructions, so that deciding what to do next becomes harder even if everything technically fits inside the context window. The source material directly identifies this as a major failure mode and notes that larger tool sets can reduce success rates.
Anthropic’s engineering advice reaches the same conclusion from the tooling side: minimal viable tool sets are preferable, tools should have minimal overlap, and ambiguous decision boundaries should be removed whenever possible.
A useful heuristic is that if two tools or two instruction blocks would confuse a new human operator, they will probably confuse the model too.
Context clash
Context clash arises when different turns, retrieved documents, or system components contribute incompatible assumptions, and the model anchors on the wrong one. The source material highlights this as a case where multi-turn sharding or conflicting inputs can materially lower accuracy.
This is why advanced context engineering prefers explicit precedence rules. If retrieved evidence disagrees with cached memory, or a newer tool result contradicts an older summary, the system should know which source wins and when to trigger a reconciliation step. Without that, conflict resolution is left to the model’s implicit heuristics, which is rarely desirable in production.
Context compression
The source material explicitly lists context compression as part of advanced context engineering, and Anthropic treats compaction as one of the core answers to long-horizon context growth.
A clean formal objective is to find a compressed representation \(z\) of history \(h\) that fits the budget \(B\) while preserving future utility:
\[z^* = \arg\max_{z : |z| \le B} U(z; h, \tau)\]
where \(U\) measures usefulness for future task \(\tau\). The point is not to create a pretty summary, but to preserve what the next step needs.
A key technical reference is
LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
by Jiang et al. (2023), which proposes coarse-to-fine prompt compression with a budget controller and shows that substantial compression can preserve much of the prompt’s utility. Put simply, Jiang et al. (2023) presents a prompt compression method that reduces token load while retaining semantic utility.
Memory hierarchies
One of the most robust advanced ideas is to stop treating context as a single flat window and instead manage it as a hierarchy: active working memory, short-term notes, and long-term external storage. This is exactly the operating-system style idea in
MemGPT: Towards LLMs as Operating Systems
by Packer et al. (2023), which introduces memory-tier management to extend effective context beyond the raw active window. In other words, it frames long-context management as movement across memory tiers rather than as naive accumulation in a single window.
This same intuition appears in recent practitioner guidance through note-taking, cached subtasks, vector-store reuse, and selective retrieval.
The practical lesson is simple: do not make the active context do the job of the entire memory system.
Progressive disclosure for exploration
Anthropic’s guidance places strong emphasis on just-in-time retrieval and progressive disclosure: the agent should often navigate references, metadata, filenames, and partial results before loading full content. That lets the system expand context only where the expected value is high.
This approach is especially effective when paired with tool-use frameworks like
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022), which shows that interleaving reasoning with environment actions can improve multi-step tasks, and
Toolformer: Language Models Can Teach Themselves to Use Tools
by Schick et al. (2023), which shows that tool invocation can be learned as part of language modeling.
Why sub-agents help
A common misconception is that multi-agent designs are powerful mainly because they parallelize work. In practice, one of their biggest benefits is contextual isolation. A specialized sub-agent can operate over a smaller, task-specific working set and then return a compact summary to a coordinator. This matches Anthropic’s recommendation to use sub-agents as a way to avoid overloading a single context window with every branch of work.
The best heuristic here is not “use more agents,” but “use a sub-agent only when it lets you partition context more cleanly than a monolithic loop would.”
Production Heuristics
Start minimal
One of the clearest design principles in the source material is to begin with the smallest context that can plausibly work, then add structure, examples, retrieval, or memory only in response to observed failure modes. Anthropic explicitly recommends starting with a minimal prompt and improving it based on failures found during testing, and the broader workflow-oriented material repeatedly ties context design to iteration and evaluation.
This is the opposite of the common “just add more prompt text” instinct. Mature systems usually improve by adding the smallest corrective signal that fixes an observed defect.
Optimize token efficiency
Advanced context engineering is often won or lost at interfaces. Tool outputs should be concise. Schemas should encode only what downstream components need. Notes should preserve commitments and references rather than raw logs. Retrieved chunks should be trimmed to their task-relevant span. The reason is straightforward: every unnecessary token consumes attention budget and increases the chance of distraction or confusion. Anthropic’s guidance is explicit that tools should promote token efficiency and that the overall goal is the smallest high-signal token set that still yields the desired outcome.
A simple objective is:
\[\max_{c} \ \frac{\mathrm{useful\ information}(c)}{|c|}\]
That ratio is not directly measurable, but it is a good north star for system design.
Separate Facts from Traces
If durable facts, current plans, tool exhaust, and historical chatter all live at the same layer, the system becomes hard to debug and easy to poison. A stronger design separates them: durable memory in one store, current state in another, and raw traces in a replay log that is not routinely injected into the active window. The source material’s distinction among state, historical context, vector-store memory, and evaluation implicitly supports this layered design.
This separation is one of the cleanest ways to reduce poisoning and clash without losing recoverability.
Evaluate the Whole System
The source material repeatedly insists that context engineering has to be measured, because stale, diluted, or inefficient context often looks reasonable by inspection while failing in execution.
That means evaluations should compare end-to-end context strategies, not isolated prompt snippets. A reasonable multi-objective score is
\[J = \alpha \cdot \text{task success} - \beta \cdot \text{latency} - \gamma \cdot \text{cost} - \delta \cdot \text{failure rate}\]
The right context is the one that improves \(J\) on real traces, not the one that merely sounds more sophisticated.
The Simplest Heuristic
Across the material, the recurring principle is unchanged: treat context as a precious, finite resource and allocate it deliberately. Anthropic states this directly, and the workflow-oriented guide arrives at the same conclusion from the angles of latency, cost, revision loops, and stale-context risk.
In practice, almost every advanced technique reduces to one of four moves:
remove low-signal tokens,
externalize state or memory,
retrieve detail only when needed,
validate that the new context actually improved behavior.
Implementation Playbook
Implementation Sequence
A strong context-engineered system is usually built in layers, not all at once. The highest-yield path is to begin with the minimum viable task loop, then add structure only where failures justify it. Recent engineering guidance repeatedly recommends starting with the simplest prompt and workflow that can plausibly work, then iterating from observed failure modes rather than preemptively stuffing more context into the window.
A practical rollout sequence looks like this:
\[\text{task prompt} \rightarrow \text{schema} \rightarrow \text{tools} \rightarrow \text{retrieval} \rightarrow \text{memory} \rightarrow \text{compaction} \rightarrow \text{evaluation loop}\]
The reason this ordering works is that each layer solves a different class of failure. The prompt gives behavioral intent, the schema makes outputs machine-usable, tools bring in dynamic context, retrieval adds external knowledge, memory preserves useful artifacts, compaction protects long-horizon performance, and evaluation tells you whether any of it actually helped.
Step 1: define the task boundary before writing the prompt
Before writing any system prompt, define exactly what the model is responsible for and what the surrounding system is responsible for. This is one of the most important hidden lessons of context engineering. Many prompt failures are really architecture failures, where the model is being asked to infer policy, retrieve facts, maintain memory, plan subtasks, and emit machine-readable outputs all from a single underspecified instruction. The more capable the surrounding system, the less the model has to infer implicitly.
A good first specification usually answers four questions:
What is the unit of work?
What information must always be present?
What information should be retrieved only when needed?
What output must downstream components receive?
A concrete structured search-planning example modeled as a multi-agent deep-research workflow appears in
Context Engineering Guide
, where a dedicated Search Planner agent is responsible only for generating a search plan from the user query, not for executing searches, revising results, or writing the final report. In that example, the sub-agent is told: “You are an expert research planner,” and its task is to break a query wrapped in
<user_query></user_query>
into exactly 2 search subtasks, each covering a different aspect or source type. The prompt requires each subtask to contain a fixed set of fields:
id
,
query
,
source_type
,
time_period
,
domain_focus
, and
priority
, and it further asks the agent to infer
start_date
and
end_date
from the current date and the chosen time period. The guide’s example input is
<user_query> What's the latest dev news from OpenAI? </user_query>
, and the example structured output includes subtasks such as
openai_latest_news
with a news-focused query and
openai_official_blog
with a web-focused query. This is a strong illustration of task-boundary design: the sub-agent has a clearly delineated task-boundary and is not asked to do everything. It is asked to produce a tightly scoped, structured planning artifact that downstream components can reliably consume.
Step 2: write the minimal complete system prompt
The system prompt should define role, objective, constraints, and output expectations using direct language and clear sectioning. Anthropic’s guidance recommends distinct sections such as background, instructions, tool guidance, and output description, and strongly warns against both brittle over-specification and vague high-level guidance.
The most useful design target is not “shortest possible prompt” but “smallest complete prompt.” In symbols, if \(c_{\text{sys}}\) is the system prompt, the aim is to minimize token cost subject to acceptable task behavior:
\[c_{\text{sys}}^* = \arg\min_{c_{\text{sys}}} |c_{\text{sys}}|
\quad \text{subject to} \quad
\mathrm{Perf}(c_{\text{sys}}) \ge \tau\]
This reflects the broader principle that context should be informative but tight, a phrase that captures much of the recent engineering consensus.
Step 3: structure the input so the model can parse the task correctly
Once the role and objective are fixed, make the input legible. Delimiters, headers, typed fields, and clearly separated sections reduce ambiguity and teach the model how to map inputs to outputs. The planning workflow in
Context Engineering Guide
uses explicit tags around the user query for exactly this reason.
This is conceptually continuous with
Language Models are Few-Shot Learners
by Brown et al. (2020), which showed that task specification emerges from how instructions, examples, and query are arranged in-context, not only from the literal wording of the instruction. The input format is therefore part of the model’s supervision signal.
Step 4: lock down the output contract early
The moment a response needs to feed another software component, output structure should become explicit. This is why schemas, typed fields, enum-like value sets, and example JSON objects are so central to production context engineering. They reduce ambiguity, increase parse success, and make failures localizable. The planning example’s use of required fields plus parser-generated schema is a very clean template for this.
A good engineering mindset is that the model is solving a constrained generation problem, not a prose-writing problem:
\[y^* = \arg\max_y p(y \mid c)
  \quad \text{such that} \quad y \in \mathcal{S}\]
where \(\mathcal{S}\) is the allowed schema. That same logic is why structured output systems often feel dramatically more reliable than free-form prompting, even when the underlying model is unchanged.
Step 5: add tools only when they remove inference burden from the model
A tool should be introduced when it lets the model stop guessing something important. Time, search, file access, calculators, execution environments, retrieval systems, and domain-specific APIs all fit this criterion. The planning example’s explicit use of current time is a textbook case: when date ranges matter, a dynamic date source is better than asking the model to infer “recent” from nowhere.
The two strongest heuristics for tool design are clarity and minimal overlap. Anthropic’s tool-design guidance is very clear that tool sets should be small, self-contained, robust, and easy to distinguish.
A useful decision test is:
\[\text{Add tool } T \text{ if } \mathrm{error\ reduction}(T) - \mathrm{complexity\ cost}(T) > 0\]
That is not a formal research metric, but it captures the systems trade-off well.
Step 6: use retrieval to keep the live context small and current
After tools, retrieval is usually the next major capability layer. The key is to avoid pushing large corpora directly into the active window. Instead, retrieve relevant fragments on demand and then curate them before insertion. This principle is central to
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
by Lewis et al. (2020), which shows how retrieval can turn external documents into model conditioning context, and it is reinforced by recent practitioner guidance that emphasizes just-in-time context loading and progressive disclosure.
A lightweight retrieval pipeline is usually:
\[\text{query} \rightarrow \text{rank} \rightarrow \text{trim} \rightarrow \text{order} \rightarrow \text{inject}\]
Skipping the trim and order steps is where many systems quietly lose quality.
Step 7: store durable artifacts, not raw traces
Once the system becomes multi-turn or long-horizon, introduce memory. But memory should usually store durable, reusable artifacts such as plans, checkpoints, user preferences, canonical summaries, or successful subqueries, rather than storing everything indiscriminately. That is the practical lesson of the vector-store reuse pattern in
Context Engineering Guide
, and it aligns with the memory-hierarchy perspective in
MemGPT: Towards LLMs as Operating Systems
by Packer et al. (2023).
A compact retrieval view of memory is:
\[m_t^* = \operatorname{TopK}_{m \in M} s(m, c_t)\]
where only the most relevant memory items are resurfaced for turn \(t\). This reinforces one of the deepest context-engineering rules: persistence is not the same thing as active context.
Step 8: introduce compaction as soon as the agent starts carrying dead weight
Long-horizon behavior eventually forces a choice: either let history accumulate and degrade the context, or compress it into a smaller representation. Anthropic’s guidance treats compaction, tool-result clearing, note-taking, and sub-agent decomposition as core long-horizon strategies, precisely because raw transcripts do not scale indefinitely.
A practical compaction checkpoint should preserve four things: current objective, unresolved questions, durable decisions, and references to where deeper detail can be reloaded. That is usually much more valuable than preserving a beautifully worded summary of everything that happened.
A directly relevant technical reference is
LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
by Jiang et al. (2023), which treats prompt compression as an optimization problem under token budgets.
Step 9: use sub-agents when they isolate context better than a single loop
Sub-agents are justified when they create cleaner local workspaces. A search planner, code reviewer, document retriever, or verifier can operate over a specialized context window and then return only a compact artifact upstream. Recent engineering guidance explicitly recommends sub-agents for long or complex tasks partly because they partition context, not just because they parallelize work.
The right question is not “can this be another agent?” but “does another agent reduce global context pollution?” If the answer is yes, the split is often worth it.
Step 10: build evaluation around failure traces, not only benchmark averages
Evaluation is the backbone of context engineering because most improvements are local and empirical. The same prompt tactic can help one task and hurt another. The same retrieval strategy can improve freshness and reduce precision. The same memory policy can improve continuity and introduce poisoning. This is why recent practitioner guidance treats evals as inseparable from context design.
A good evaluation loop is usually:
\[\text{failure trace} \rightarrow \text{hypothesized context defect} \rightarrow \text{context change} \rightarrow \text{retest}\]
That workflow is much more informative than endlessly rewriting prompts without a grounded theory of failure.
Design checklist
When designing a context-engineered system, the following questions usually matter more than any single clever prompt trick.
Does the system prompt define role, boundary, and output expectations clearly enough to avoid hidden assumptions?
Is the input structured so the model can distinguish instructions, user data, retrieved evidence, and tool outputs?
Is the output contract explicit enough for downstream systems to consume reliably?
Do tools remove genuine inference burden, or do they merely add more choice complexity?
Is retrieved information being curated before insertion, rather than dumped wholesale into context?
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
by Lewis et al. (2020) shows retrieved evidence improves generation only when it is incorporated meaningfully into the model’s conditioning context.
Are memory and state separated from raw trace history?
MemGPT: Towards LLMs as Operating Systems
by Packer et al. (2023) argues for memory tiers instead of treating one flat context window as the whole system.
Does the system compact history before distraction, confusion, or clash dominate behavior?
Lost in the Middle: How Language Models Use Long Contexts
by Liu et al. (2024) shows that long-context performance depends strongly on information placement and retrieval conditions;
LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
by Jiang et al. (2023) shows prompt compression can preserve useful signal under token limits.
Are context changes being evaluated on real traces with quality, cost, latency, and failure rate all visible?
Final synthesis
The central lesson of context engineering is that model capability is only part of the story. The rest is the design of the information environment in which the model operates. The strongest systems do not merely “prompt the model well.” They decide what the model should know now, what it should retrieve later, what it should remember, what it should forget, what form its outputs should take, and how those choices will be tested. That is why the field has moved from prompt engineering toward context engineering as the more complete design language for modern AI systems.
At its most compact, the discipline reduces to one question:
What is the smallest, clearest, highest-signal context that makes the next step go right?
That is the core of context engineering.
References
Context Engineering Blogs/Guides
Effective context engineering for AI agents
Writing effective tools for AI agents
Building Effective AI Agents
Context Engineering Guide
How Long Contexts Fail
How to Fix Your Context
Prompt Engineering Guide
Prompting, in-context learning, and reasoning
Language Models are Few-Shot Learners
by Brown et al. (2020)
Attention Is All You Need
by Vaswani et al. (2017)
Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
by Wei et al. (2022)
Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models
by Wang et al. (2023)
Tree of Thoughts: Deliberate Problem Solving with Large Language Models
by Yao et al. (2023)
Self-Refine: Iterative Refinement with Self-Feedback
by Madaan et al. (2023)
Active Prompting with Chain-of-Thought for Large Language Models
by Diao et al. (2023)
Demystifying Chains, Trees, and Graphs of Thoughts
by Besta et al. (2024)
Retrieval, long-context behavior, and compression
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
by Lewis et al. (2020)
Lost in the Middle: How Language Models Use Long Contexts
by Liu et al. (2024)
LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
by Jiang et al. (2023)
Context Rot: How Increasing Input Tokens Impacts LLM Performance
by Hong et al. (2025)
Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding
by Zhang et al. (2024)
Tool use, acting, and execution
ReAct: Synergizing Reasoning and Acting in Language Models
by Yao et al. (2022)
Toolformer: Language Models Can Teach Themselves to Use Tools
by Schick et al. (2023)
Memory, agents, and long-horizon systems
MemGPT: Towards LLMs as Operating Systems
by Packer et al. (2023)
Generative Agents: Interactive Simulacra of Human Behavior
by Park et al. (2023)
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledContextEngineering,
  title   = {Context Engineering},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}