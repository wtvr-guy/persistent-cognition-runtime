# Prior work and research approach

This scaffold inherits architectural questions from the source project's supplied neuroscience and similar-project research notes. Those notes contain historical assessments and should not be treated as a current feature inventory.

## Reading map

| Reference | Question for this project |
| --- | --- |
| [Soar](https://soar.eecs.umich.edu/) | How should situation, goals, operators, and episodic/semantic memory be separated? |
| [ACT-R](https://act-r.psy.cmu.edu/) | What can module and buffer boundaries teach us about bounded active state? |
| [LIDA](https://ccrg.cs.memphis.edu/) | How should perception, attention, and action selection be separated? |
| [MemGPT](https://arxiv.org/abs/2310.08560) | How can external memory support limited model context? |
| [CoALA](https://arxiv.org/abs/2309.02427) | Which memory and action categories clarify language-agent architecture? |
| [AIOS](https://arxiv.org/abs/2403.16971) | How should scheduling and resource services be separated from agent computation? |
| [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | Which checkpoint and recovery contracts are useful for durable execution? |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | How should observation, reflection, and planning be evaluated? |
| [Voyager](https://arxiv.org/abs/2305.16291) | How can verified procedures become reusable skills? |

This is a reading map, not a current comparative feature audit or a novelty claim.

## Neuroscience as engineering inspiration

Cue-driven retrieval, working-memory limits, complementary learning systems, episode segmentation, and replay suggest hypotheses for derived memory and attention.

An immutable software ledger is not a biological memory mechanism. Biological resemblance is not an acceptance criterion. Retain a mechanism only when a frozen experiment demonstrates useful behavior at acceptable cost.

## Positioning

External memory, stateless inference, durable workflows, and cognitive architectures all have substantial prior work. The research question here is whether their composition around durable system state, narrow disposable reasoning, explicit causal evidence, and measured resource admission yields useful continuous operation on modest hardware.
