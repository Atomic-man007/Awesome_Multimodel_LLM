# Awesome-Multimodal-LLM


<font size=6><center><big><b> [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) </b></big></center></font>

✨✨✨ Behold our meticulously curated trove of Multimodal Large Language Models (MLLM) resources! 📚🔍 Feast your eyes on an assortment of datasets, techniques for tuning multimodal instructions, methods for multimodal in-context learning, approaches for multimodal chain-of-thought, visual reasoning aided by gargantuan language models, foundational models, and much more. 🌟🔥

✨✨✨ This compilation shall forever stay in sync with the vanguard of breakthroughs in the realm of MLLM. 🔄 We are committed to its perpetual evolution, ensuring that you never miss out on the latest developments. 🚀💡

✨✨✨ And hold your breath, for we are diligently crafting a survey paper on latest LLM & MLLM, which shall soon grace the world with its wisdom. Stay tuned for its grand debut! 🎉📑

![](/resources/LLMS.png)
---

<font size=5><center><b> Table of Contents </b> </center></font>

- [Awesome Papers](#awesome-papers)
  - [Multimodal Instruction Tuning](#multimodal-instruction-tuning)
  - [Multimodal In-Context Learning](#multimodal-in-context-learning)
  - [Multimodal Chain-of-Thought](#multimodal-chain-of-thought)
  - [LLM-Aided Visual Reasoning](#llm-aided-visual-reasoning)
  - [Foundation Models](#foundation-models)
  - [Others](#others)
- [Awesome Datasets](#awesome-datasets)
  - [Datasets of Pre-Training for Alignment](#datasets-of-pre-training-for-alignment)
  - [Datasets of Multimodal Instruction Tuning](#datasets-of-multimodal-instruction-tuning)
  - [Datasets of In-Context Learning](#datasets-of-in-context-learning)
  - [Datasets of Multimodal Chain-of-Thought](#datasets-of-multimodal-chain-of-thought)
  - [RLHFdataset](#RLHFdataset)


* [Updates](#updates)
* [Survey Introduction](#survey-introduction)
* [Markups](#markups)
* [Table of Contents](#table-of-contents)
* [Related Surveys for LLMs Evaluation](#related-surveys-for-LLMs-evaluation)
* [Papers](#papers)
    * [Knowledge and Capability Evaluation](#booksknowledge-and-capability-evaluation)
        * [Question Answering](#question-answering)
        * [Knowledge Completion](#knowledge-completion)
        * [Reasoning](#reasoning)
            * [Commonsense Reasoning](#commonsense-reasoning)
            * [Logical Reasoning](#logical-reasoning)
            * [Multi-hop Reasoning](#multi-hop-reasoning)
            * [Mathematical Reasoning](#mathematical-reasoning)
        * [Tool Learning](#tool-learning)
      
    * [Alignment Evaluation](#triangular_ruleralignment-evaluation)
        * [Ethics and Morality](#ethics-and-morality)
        * [Bias](#bias)
        * [Toxicity](#toxicity)
        * [Truthfulness](#truthfulness)
      * [General Alignment Evaluation](#general-alignment-evaluation)
      
    * [Safety Evaluation](#closed_lock_with_keysafety-evaluation)
        * [Robustness Evaluation](#robustness-evaluation)
        * [Risk Evaluation](#risk-evaluation)
            * [Evaluating LLMs Behaviors](#evaluating-llms-behaviors)
            * [Evaluating LLMs as Agents](#evaluating-llms-as-agents)
      
    * [Specialized LLMs Evaluation](#syringewoman_judgecomputermoneybagspecialized-llms-evaluation)
        * [Biology and Medicine](#biology-and-medicine)
        * [Education](#education)
        * [Legislation](#legislation)
        * [Computer Science](#computer-science)
        * [Finance](#finance)
      
    * [Evaluation Organization](#earth_americasevaluation-organization)
        * [Benchmarks for NLU and NLG](#benchmarks-for-nlu-and-nlg)
        * [Benchmarks for Knowledge and Reasoning](#benchmarks-for-knowledge-and-reasoning)
        * [Benchmark for Holistic Evaluation](#benchmark-for-holistic-evaluation)
      
    * [LLM Leaderboards](#llm-leaderboards)
      
----
## LLM Learning MindMap
![](/resources/LLM_Learning_MindMap.png)

![](/resources/llm.png)

---
## Trending LLM Projects
- [llm-course](https://github.com/mlabonne/llm-course) - Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.
- [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) - a high-quality sparse mixture of experts model (SMoE) with open weights.
- [promptbase](https://github.com/microsoft/promptbase) - All things prompt engineering.
- [ollama](https://github.com/jmorganca/ollama) - Get up and running with Llama 2 and other large language models locally.
- [Devika](https://github.com/stitionai/devika) Devin alternate SDE LLM 
- [anything-llm](https://github.com/Mintplex-Labs/anything-llm) - A private ChatGPT to chat with anything!
- [phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) - a 2.7 billion-parameter language model that demonstrates outstanding reasoning and language understanding capabilities, showcasing state-of-the-art performance among base language models with less than 13 billion parameters.

## Practical Guides for Prompting (Helpful)

- **OpenAI Cookbook**. [Blog](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)
- **Prompt Engineering**. [Blog](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- **ChatGPT Prompt Engineering for Developers!** [Course](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
  
## High-quality generation


- [2023/10] **Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond** *Liang Chen et al. arXiv.* [[paper](https://arxiv.org/abs/2310.02071)] [[code](https://github.com/PKUnlp-icler/PCA-EVAL)]
  - This work proposes PCA-EVAL, which benchmarks embodied decision making via MLLM-based End-to-End method and LLM-based Tool-Using methods from Perception, Cognition and Action Levels.
- [2023/08] **A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity.** *Yejin Bang et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2302.04023)]
  - This work evaluates the multitask, multilingual and multimodal aspects of ChatGPT using 21 data sets covering 8 different common NLP application tasks.
- [2023/06] **LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models.** *Yen-Ting Lin et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2305.13711)]
  - The LLM-EVAL method evaluates multiple dimensions of evaluation, such as content, grammar, relevance, and appropriateness.
- [2023/04] **Is ChatGPT a Highly Fluent Grammatical Error Correction System? A Comprehensive Evaluation.** *Tao Fang et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2304.01746)]
  - The results of evaluation demonstrate that ChatGPT has excellent error detection capabilities and can freely correct errors to make the corrected sentences very fluent. Additionally, its performance in non-English and low-resource settings highlights its potential in multilingual GEC tasks.

## Deep understanding

- [2023/06] **Clever Hans or Neural Theory of Mind? Stress Testing Social Reasoning in Large Language Models.** *Natalie Shapira et al. arXiv.* [[paper](https://doi.org/10.48550/arXiv.2305.14763)]
  - LLMs exhibit certain theory of mind abilities, but this behavior is far from being robust.
- [2022/08] **Inferring Rewards from Language in Context.** *Jessy Lin et al. ACL.* [[paper](https://doi.org/10.18653/v1/2022.acl-long.585)]
  - This work presents a model that infers rewards from language and predicts optimal actions in unseen environment.
- [2021/10] **Theory of Mind Based Assistive Communication in Complex Human Robot Cooperation.** *Moritz C. Buehler et al. arXiv.* [[paper](https://arxiv.org/abs/2109.01355)]
  - This work designs an agent Sushi with an understanding of the human during interaction.

## Memory capability

## Raising the length limit of Transformers

- [2023/10] **MemGPT: Towards LLMs as Operating Systems.** *Charles Packer (UC Berkeley) et al. arXiv.* [[paper](https://arxiv.org/abs/2310.08560)] [[project page](https://memgpt.ai/)] [[code](https://github.com/cpacker/MemGPT)] [[dataset](https://huggingface.co/MemGPT)]
- [2023/05] **Randomized Positional Encodings Boost Length Generalization of Transformers.** *Anian Ruoss (DeepMind) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.16843)] [[code](https://github.com/google-deepmind/randomized_positional_encodings)]
- [2023-03] **CoLT5: Faster Long-Range Transformers with Conditional Computation.** *Joshua Ainslie (Google Research) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.09752)]
- [2022/03] **Efficient Classification of Long Documents Using Transformers.** *Hyunji Hayley Park (Illinois University) et al. arXiv.* [[paper](https://arxiv.org/abs/2203.11258)] [[code](https://github.com/amazon-science/efficient-longdoc-classification)]
- [2021/12] **LongT5: Efficient Text-To-Text Transformer for Long Sequences.** *Mandy Guo (Google Research) et al. arXiv.* [[paper](https://arxiv.org/abs/2112.07916)] [[code](https://github.com/google-research/longt5)]
- [2019/10] **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.** *Michael Lewis (Facebook AI) et al. arXiv.* [[paper](https://arxiv.org/abs/1910.13461)] [[code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bart)]

###### Summarizing memory

- [2023/10] **Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading** *Howard Chen (Princeton University) et al. arXiv.* [[paper](https://arxiv.org/abs/2310.05029)]
- [2023/09] **Empowering Private Tutoring by Chaining Large Language Models** *Yulin Chen (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2309.08112)]
- [2023/08] **ExpeL: LLM Agents Are Experiential Learners.** *Andrew Zhao (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.10144)] [[code](https://github.com/Andrewzh112/ExpeL)]
- [2023/08] **ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate.** *Chi-Min Chan (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.07201)] [[code](https://github.com/thunlp/ChatEval)]
- [2023/05] **MemoryBank: Enhancing Large Language Models with Long-Term Memory.** *Wanjun Zhong (Harbin Institute of Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.10250)] [[code](https://github.com/zhongwanjun/memorybank-siliconfriend)]
- [2023/04] **Generative Agents: Interactive Simulacra of Human Behavior.** *Joon Sung Park (Stanford University) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.03442)] [[code](https://github.com/joonspk-research/generative_agents)]
- [2023/04] **Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System.** *Xinnian Liang (Beihang University) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.13343)] [[code](https://github.com/wbbeyourself/scm4llms)]
- [2023/03] **Reflexion: Language Agents with Verbal Reinforcement Learning.** *Noah Shinn (Northeastern University) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.11366)] [[code](https://github.com/noahshinn024/reflexion)]
- [2023/05] **RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text.** *Wangchunshu Zhou (AIWaves) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.13304.pdf)] [[code](https://github.com/aiwaves-cn/RecurrentGPT)]


## Compressing memories with vectors or data structures

- [2023/07] **Communicative Agents for Software Development.** *Chen Qian (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.07924)] [[code](https://github.com/openbmb/chatdev)]
- [2023/06] **ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory.** *Chenxu Hu (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.03901)] [[code](https://github.com/huchenxucs/ChatDB)]
- [2023/05] **Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory.** *Xizhou Zhu (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.17144)] [[code](https://github.com/OpenGVLab/GITM)]
- [2023/05] **RET-LLM: Towards a General Read-Write Memory for Large Language Models.** *Ali Modarressi (LMU Munich) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14322)] [[code](https://github.com/tloen/alpaca-lora)]
- [2023/05] **RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text.** *Wangchunshu Zhou (AIWaves) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.13304.pdf)] [[code](https://github.com/aiwaves-cn/RecurrentGPT)]

## Memory retrieval

- [2023/08] **Memory Sandbox: Transparent and Interactive Memory Management for Conversational Agents.** *Ziheng Huang (University of California—San Diego) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.01542)]
- [2023/08] **AgentSims: An Open-Source Sandbox for Large Language Model Evaluation.** *Jiaju Lin (PTA Studio) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.04026)] [[project page](https://www.agentsims.com/)] [[code](https://github.com/py499372727/AgentSims/)]
- [2023/06] **ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory.** *Chenxu Hu (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.03901)] [[code](https://github.com/huchenxucs/ChatDB)]
- [2023/05] **MemoryBank: Enhancing Large Language Models with Long-Term Memory.** *Wanjun Zhong (Harbin Institute of Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.10250)] [[code](https://github.com/zhongwanjun/memorybank-siliconfriend)]
- [2023/04] **Generative Agents: Interactive Simulacra of Human Behavior.** *Joon Sung Park (Stanford) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.03442)] [[code](https://github.com/joonspk-research/generative_agents)]
- [2023/05] **RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text.** *Wangchunshu Zhou (AIWaves) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.13304.pdf)] [[code](https://github.com/aiwaves-cn/RecurrentGPT)]

## Awesome Papers

## Multimodal Instruction Tuning
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/mbzuai-oryx/Video-ChatGPT.svg?style=social&label=Star) <br> [**Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models**](https://arxiv.org/pdf/2306.05424.pdf) <br> | arXiv | 2023-06-08 | [Github](https://github.com/mbzuai-oryx/Video-ChatGPT) | [Demo](https://www.ival-mbzuai.com/video-chatgpt) |
| ![Star](https://img.shields.io/github/stars/Luodian/Otter.svg?style=social&label=Star) <br> [**MIMIC-IT: Multi-Modal In-Context Instruction Tuning**](https://arxiv.org/pdf/2306.05425.pdf) <br> | arXiv | 2023-06-08 | [Github](https://github.com/Luodian/Otter) | [Demo](https://otter.cliangyu.com/) |
| [**M<sup>3</sup>IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning**](https://arxiv.org/pdf/2306.04387.pdf) | arXiv | 2023-06-07 | - | - | 
| ![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social&label=Star) <br> [**Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**](https://arxiv.org/pdf/2306.02858.pdf) <br> | arXiv | 2023-06-05 | [Github](https://github.com/DAMO-NLP-SG/Video-LLaMA) | [Demo](https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA) |
| ![Star](https://img.shields.io/github/stars/microsoft/LLaVA-Med.svg?style=social&label=Star) <br> [**LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day**](https://arxiv.org/pdf/2306.00890.pdf) <br> | arXiv | 2023-06-01 | [Github](https://github.com/microsoft/LLaVA-Med) | - |
| ![Star](https://img.shields.io/github/stars/StevenGrove/GPT4Tools.svg?style=social&label=Star) <br> [**GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**](https://arxiv.org/pdf/2305.18752.pdf) <br> | arXiv | 2023-05-30 | [Github](https://github.com/StevenGrove/GPT4Tools) | [Demo](https://huggingface.co/spaces/stevengrove/GPT4Tools) | 
| ![Star](https://img.shields.io/github/stars/yxuansu/PandaGPT.svg?style=social&label=Star) <br> [**PandaGPT: One Model To Instruction-Follow Them All**](https://arxiv.org/pdf/2305.16355.pdf) <br> | arXiv | 2023-05-25 | [Github](https://github.com/yxuansu/PandaGPT) | [Demo](https://huggingface.co/spaces/GMFTBY/PandaGPT) | 
| ![Star](https://img.shields.io/github/stars/joez17/ChatBridge.svg?style=social&label=Star) <br> [**ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst**](https://arxiv.org/pdf/2305.16103.pdf) <br> | arXiv | 2023-05-25 | [Github](https://github.com/joez17/ChatBridge) | - | 
| ![Star](https://img.shields.io/github/stars/luogen1996/LaVIN.svg?style=social&label=Star) <br> [**Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models**](https://arxiv.org/pdf/2305.15023.pdf) <br> | arXiv | 2023-05-24 | [Github](https://github.com/luogen1996/LaVIN) | Local Demo |
| ![Star](https://img.shields.io/github/stars/OptimalScale/DetGPT.svg?style=social&label=Star) <br> [**DetGPT: Detect What You Need via Reasoning**](https://arxiv.org/pdf/2305.14167.pdf) <br> | arXiv | 2023-05-23 | [Github](https://github.com/OptimalScale/DetGPT) | [Demo](https://d3c431c0c77b1d9010.gradio.live/) | 
| ![Star](https://img.shields.io/github/stars/OpenGVLab/VisionLLM.svg?style=social&label=Star) <br> [**VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks**](https://arxiv.org/pdf/2305.11175.pdf) <br> | arXiv | 2023-05-18 | [Github](https://github.com/OpenGVLab/VisionLLM) | [Demo](https://igpt.opengvlab.com/) |
| ![Star](https://img.shields.io/github/stars/THUDM/VisualGLM-6B.svg?style=social&label=Star) <br> **VisualGLM-6B** <br> | - | 2023-05-17 | [Github](https://github.com/THUDM/VisualGLM-6B) | Local Demo |
| ![Star](https://img.shields.io/github/stars/xiaoman-zhang/PMC-VQA.svg?style=social&label=Star) <br> [**PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering**](https://arxiv.org/pdf/2305.10415.pdf) <br> | arXiv | 2023-05-17 | [Github](https://github.com/xiaoman-zhang/PMC-VQA) | - | 
| ![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star) <br> [**InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**](https://arxiv.org/pdf/2305.06500.pdf) <br> | arXiv | 2023-05-11 | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | Local Demo |
| ![Star](https://img.shields.io/github/stars/OpenGVLab/Ask-Anything.svg?style=social&label=Star) <br> [**VideoChat: Chat-Centric Video Understanding**](https://arxiv.org/pdf/2305.06355.pdf) <br> | arXiv | 2023-05-10 | [Github](https://github.com/OpenGVLab/Ask-Anything) | [Demo](https://ask.opengvlab.com/) |
| ![Star](https://img.shields.io/github/stars/open-mmlab/Multimodal-GPT.svg?style=social&label=Star) <br> [**MultiModal-GPT: A Vision and Language Model for Dialogue with Humans**](https://arxiv.org/pdf/2305.04790.pdf) <br> | arXiv | 2023-05-08 | [Github](https://github.com/open-mmlab/Multimodal-GPT) | [Demo](https://mmgpt.openmmlab.org.cn/) |
| ![Star](https://img.shields.io/github/stars/phellonchen/X-LLM.svg?style=social&label=Star) <br> [**X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages**](https://arxiv.org/pdf/2305.04160.pdf) <br> | arXiv | 2023-05-07 | [Github](https://github.com/phellonchen/X-LLM) | - | 
| ![Star](https://img.shields.io/github/stars/ZrrSkywalker/LLaMA-Adapter.svg?style=social&label=Star) <br> [**LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model**](https://arxiv.org/pdf/2304.15010.pdf) <br> | arXiv | 2023-04-28 | [Github](https://github.com/ZrrSkywalker/LLaMA-Adapter) | [Demo](http://llama-adapter.opengvlab.com/) | 
| ![Star](https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl.svg?style=social&label=Star) <br> [**mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality**](https://arxiv.org/pdf/2304.14178.pdf) <br> | arXiv | 2023-04-27 | [Github](https://github.com/X-PLUG/mPLUG-Owl) | [Demo](https://huggingface.co/spaces/MAGAer13/mPLUG-Owl) |
| ![Star](https://img.shields.io/github/stars/Vision-CAIR/MiniGPT-4.svg?style=social&label=Star) <br> [**MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**](https://arxiv.org/pdf/2304.10592.pdf) <br> | arXiv | 2023-04-20 | [Github](https://github.com/Vision-CAIR/MiniGPT-4) | - |
| ![Star](https://img.shields.io/github/stars/haotian-liu/LLaVA.svg?style=social&label=Star) <br> [**Visual Instruction Tuning**](https://arxiv.org/pdf/2304.08485.pdf) <br> | arXiv | 2023-04-17 | [GitHub](https://github.com/haotian-liu/LLaVA) | [Demo](https://llava.hliu.cc/) |
| ![Star](https://img.shields.io/github/stars/ZrrSkywalker/LLaMA-Adapter.svg?style=social&label=Star) <br> [**LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**](https://arxiv.org/pdf/2303.16199.pdf) <br> | arXiv | 2023-03-28 | [Github](https://github.com/ZrrSkywalker/LLaMA-Adapter) | [Demo](https://huggingface.co/spaces/csuhan/LLaMA-Adapter) |
| [**MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning**](https://arxiv.org/pdf/2212.10773.pdf) | arXiv | 2022-12-21 | - | - | 


## Multimodal In-Context Learning
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/Luodian/Otter.svg?style=social&label=Star) <br> [**MIMIC-IT: Multi-Modal In-Context Instruction Tuning**](https://arxiv.org/pdf/2306.05425.pdf) <br> | arXiv | 2023-06-08 | [Github](https://github.com/Luodian/Otter) | [Demo](https://otter.cliangyu.com/) |
| ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star) <br> [**Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**](https://arxiv.org/pdf/2304.09842.pdf) <br> | arXiv | 2023-04-19 | [Github](https://github.com/lupantech/chameleon-llm) | [Demo](https://chameleon-llm.github.io/) | 
| ![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star) <br> [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace**](https://arxiv.org/pdf/2303.17580.pdf) <br> | arXiv | 2023-03-30 | [Github](https://github.com/microsoft/JARVIS) | [Demo](https://huggingface.co/spaces/microsoft/HuggingGPT) | 
| ![Star](https://img.shields.io/github/stars/microsoft/MM-REACT.svg?style=social&label=Star) <br> [**MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**](https://arxiv.org/pdf/2303.11381.pdf) <br> | arXiv | 2023-03-20 | [Github](https://github.com/microsoft/MM-REACT) | [Demo](https://huggingface.co/spaces/microsoft-cognitive-service/mm-react) |
| ![Star](https://img.shields.io/github/stars/MILVLG/prophet.svg?style=social&label=Star) <br> [**Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering**](https://arxiv.org/pdf/2303.01903.pdf) <br> | CVPR | 2023-03-03 | [Github](https://github.com/MILVLG/prophet) | - |
| ![Star](https://img.shields.io/github/stars/allenai/visprog.svg?style=social&label=Star) <br> [**Visual Programming: Compositional visual reasoning without training**](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf) <br> | CVPR | 2022-11-18 | [Github](https://github.com/allenai/visprog) | Local Demo | 
| ![Star](https://img.shields.io/github/stars/microsoft/PICa.svg?style=social&label=Star) <br> [**An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA**](https://ojs.aaai.org/index.php/AAAI/article/download/20215/19974) <br> | AAAI | 2022-06-28 | [Github](https://github.com/microsoft/PICa) | - |
| ![Star](https://img.shields.io/github/stars/mlfoundations/open_flamingo.svg?style=social&label=Star) <br> [**Flamingo: a Visual Language Model for Few-Shot Learning**](https://arxiv.org/pdf/2204.14198.pdf) <br> | NeurIPS | 2022-04-29 | [Github](https://github.com/mlfoundations/open_flamingo) | [Demo](https://huggingface.co/spaces/dhansmair/flamingo-mini-cap) | 
| [**Multimodal Few-Shot Learning with Frozen Language Models**](https://arxiv.org/pdf/2106.13884.pdf) | NeurIPS | 2021-06-25 | - | - |


## Multimodal Chain-of-Thought
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/yu-rp/VisualPerceptionToken.svg?style=social&label=Star) <br> [**Introducing Visual Perception Token into Multimodal Large Language Model**](https://arxiv.org/pdf/2502.17425.pdf) <br> | arXiv | 2025-02-24 | [Github](https://github.com/yu-rp/VisualPerceptionToken) | - | 
| ![Star](https://img.shields.io/github/stars/EmbodiedGPT/EmbodiedGPT_Pytorch.svg?style=social&label=Star) <br> [**EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought**](https://arxiv.org/pdf/2305.15021.pdf) <br> | arXiv | 2023-05-24 | [Github](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) | - | 
| [**Let’s Think Frame by Frame: Evaluating Video Chain of Thought with Video Infilling and Prediction**](https://arxiv.org/pdf/2305.13903.pdf) | arXiv | 2023-05-23 | - | - |
| ![Star](https://img.shields.io/github/stars/ttengwang/Caption-Anything.svg?style=social&label=Star) <br> [**Caption Anything: Interactive Image Description with Diverse Multimodal Controls**](https://arxiv.org/pdf/2305.02677.pdf) <br> | arXiv | 2023-05-04 | [Github](https://github.com/ttengwang/Caption-Anything) | [Demo](https://huggingface.co/spaces/TencentARC/Caption-Anything) |
| [**Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings**](https://arxiv.org/pdf/2305.02317.pdf) | arXiv | 2023-05-03 | [Coming soon](https://github.com/dannyrose30/VCOT) | - |
| ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star) <br> [**Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**](https://arxiv.org/pdf/2304.09842.pdf) <br> | arXiv | 2023-04-19 | [Github](https://github.com/lupantech/chameleon-llm) | [Demo](https://chameleon-llm.github.io/) | 
| [**Chain of Thought Prompt Tuning in Vision Language Models**](https://arxiv.org/pdf/2304.07919.pdf) | arXiv | 2023-04-16 | [Coming soon]() | - |
| ![Star](https://img.shields.io/github/stars/microsoft/MM-REACT.svg?style=social&label=Star) <br> [**MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**](https://arxiv.org/pdf/2303.11381.pdf) <br> | arXiv | 2023-03-20 | [Github](https://github.com/microsoft/MM-REACT) | [Demo](https://huggingface.co/spaces/microsoft-cognitive-service/mm-react) |
| ![Star](https://img.shields.io/github/stars/microsoft/TaskMatrix.svg?style=social&label=Star) <br> [**Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models**](https://arxiv.org/pdf/2303.04671.pdf) <br> | arXiv | 2023-03-08 | [Github](https://github.com/microsoft/TaskMatrix) | [Demo](https://huggingface.co/spaces/microsoft/visual_chatgpt) |
| ![Star](https://img.shields.io/github/stars/amazon-science/mm-cot.svg?style=social&label=Star) <br> [**Multimodal Chain-of-Thought Reasoning in Language Models**](https://arxiv.org/pdf/2302.00923.pdf) <br> | arXiv | 2023-02-02 | [Github](https://github.com/amazon-science/mm-cot) | - |
| ![Star](https://img.shields.io/github/stars/allenai/visprog.svg?style=social&label=Star) <br> [**Visual Programming: Compositional visual reasoning without training**](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf) <br> | CVPR | 2022-11-18 | [Github](https://github.com/allenai/visprog) | Local Demo | 
| ![Star](https://img.shields.io/github/stars/lupantech/ScienceQA.svg?style=social&label=Star) <br> [**Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering**](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf) <br> | NeurIPS | 2022-09-20 | [Github](https://github.com/lupantech/ScienceQA) | - |


## LLM-Aided Visual Reasoning
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/StevenGrove/GPT4Tools.svg?style=social&label=Star) <br> [**GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**](https://arxiv.org/pdf/2305.18752.pdf) <br> | arXiv | 2023-05-30 | [Github](https://github.com/StevenGrove/GPT4Tools) | [Demo](https://c60eb7e9400930f31b.gradio.live/) | 
| ![Star](https://img.shields.io/github/stars/weixi-feng/LayoutGPT.svg?style=social&label=Star) <br> [**LayoutGPT: Compositional Visual Planning and Generation with Large Language Models**](https://arxiv.org/pdf/2305.15393.pdf) <br> | arXiv | 2023-05-24 | [Github](https://github.com/weixi-feng/LayoutGPT) | - |
| ![Star](https://img.shields.io/github/stars/Hxyou/IdealGPT.svg?style=social&label=Star) <br> [**IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models**](https://arxiv.org/pdf/2305.14985.pdf) <br> | arXiv | 2023-05-24 | [Github](https://github.com/Hxyou/IdealGPT) | Local Demo | 
| ![Star](https://img.shields.io/github/stars/ttengwang/Caption-Anything.svg?style=social&label=Star) <br> [**Caption Anything: Interactive Image Description with Diverse Multimodal Controls**](https://arxiv.org/pdf/2305.02677.pdf) <br> | arXiv | 2023-05-04 | [Github](https://github.com/ttengwang/Caption-Anything) | [Demo](https://huggingface.co/spaces/TencentARC/Caption-Anything) |
| ![Star](https://img.shields.io/github/stars/lupantech/chameleon-llm.svg?style=social&label=Star) <br> [**Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**](https://arxiv.org/pdf/2304.09842.pdf) <br> | arXiv | 2023-04-19 | [Github](https://github.com/lupantech/chameleon-llm) | [Demo](https://chameleon-llm.github.io/) | 
| ![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star) <br> [**HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace**](https://arxiv.org/pdf/2303.17580.pdf) <br> | arXiv | 2023-03-30 | [Github](https://github.com/microsoft/JARVIS) | [Demo](https://huggingface.co/spaces/microsoft/HuggingGPT) | 
| ![Star](https://img.shields.io/github/stars/microsoft/MM-REACT.svg?style=social&label=Star) <br> [**MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**](https://arxiv.org/pdf/2303.11381.pdf) <br> | arXiv | 2023-03-20 | [Github](https://github.com/microsoft/MM-REACT) | [Demo](https://huggingface.co/spaces/microsoft-cognitive-service/mm-react) |
| ![Star](https://img.shields.io/github/stars/cvlab-columbia/viper.svg?style=social&label=Star) <br> [**ViperGPT: Visual Inference via Python Execution for Reasoning**](https://arxiv.org/pdf/2303.08128.pdf) <br> | arXiv | 2023-03-14 | [Github](https://github.com/cvlab-columbia/viper) | Local Demo | 
| ![Star](https://img.shields.io/github/stars/Vision-CAIR/ChatCaptioner.svg?style=social&label=Star) <br> [**ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions**](https://arxiv.org/pdf/2303.06594.pdf) <br> | arXiv | 2023-03-12 | [Github](https://github.com/Vision-CAIR/ChatCaptioner) | Local Demo |
| ![Star](https://img.shields.io/github/stars/microsoft/TaskMatrix.svg?style=social&label=Star) <br> [**Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models**](https://arxiv.org/pdf/2303.04671.pdf) <br> | arXiv | 2023-03-08 | [Github](https://github.com/microsoft/TaskMatrix) | [Demo](https://huggingface.co/spaces/microsoft/visual_chatgpt) |
| ![Star](https://img.shields.io/github/stars/ZrrSkywalker/CaFo.svg?style=social&label=Star) <br> [**Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners**](https://arxiv.org/pdf/2303.02151.pdf) <br> | CVPR | 2023-03-03 | [Github](https://github.com/ZrrSkywalker/CaFo) | - |
| ![Star](https://img.shields.io/github/stars/yangyangyang127/PointCLIP_V2.svg?style=social&label=Star) <br> [**PointCLIP V2: Adapting CLIP for Powerful 3D Open-world Learning**](https://arxiv.org/pdf/2211.11682.pdf) <br> | CVPR | 2022-11-21 | [Github](https://github.com/yangyangyang127/PointCLIP_V2) | - |
| ![Star](https://img.shields.io/github/stars/allenai/visprog.svg?style=social&label=Star) <br> [**Visual Programming: Compositional visual reasoning without training**](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf) <br> | CVPR | 2022-11-18 | [Github](https://github.com/allenai/visprog) | Local Demo | 
| ![Star](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star) <br> [**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language**](https://arxiv.org/pdf/2204.00598.pdf) <br> | arXiv | 2022-04-01 | [Github](https://github.com/google-research/google-research/tree/master/socraticmodels) | - |


## Foundation Models
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/VPGTrans/VPGTrans.svg?style=social&label=Star) <br> [**Transfer Visual Prompt Generator across LLMs**](https://arxiv.org/pdf/2305.01278.pdf) <br> | arXiv | 2023-05-02 | [Github](https://github.com/VPGTrans/VPGTrans) | [Demo](https://3fc7715dbc44234a7f.gradio.live/) | 
| [**GPT-4 Technical Report**](https://arxiv.org/pdf/2303.08774.pdf) | arXiv | 2023-03-15 | - | - |
| [**PaLM-E: An Embodied Multimodal Language Model**](https://arxiv.org/pdf/2303.03378.pdf) | arXiv | 2023-03-06 | - | [Demo](https://palm-e.github.io/#demo) | 
| ![Star](https://img.shields.io/github/stars/NVlabs/prismer.svg?style=social&label=Star) <br> [**Prismer: A Vision-Language Model with An Ensemble of Experts**](https://arxiv.org/pdf/2303.02506.pdf) <br> | arXiv | 2023-03-04 | [Github](https://github.com/NVlabs/prismer) | [Demo](https://huggingface.co/spaces/lorenmt/prismer) |
| ![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star) <br> [**Language Is Not All You Need: Aligning Perception with Language Models**](https://arxiv.org/pdf/2302.14045.pdf) <br> | arXiv | 2023-02-27 | [Github](https://github.com/microsoft/unilm) | - |
| ![Star](https://img.shields.io/github/stars/salesforce/LAVIS.svg?style=social&label=Star) <br> [**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**](https://arxiv.org/pdf/2301.12597.pdf) <br> | arXiv | 2023-01-30 | [Github](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | [Demo](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb) | 
| ![Star](https://img.shields.io/github/stars/vimalabs/VIMA.svg?style=social&label=Star) <br> [**VIMA: General Robot Manipulation with Multimodal Prompts**](https://arxiv.org/pdf/2210.03094.pdf) <br> | ICML | 2022-10-06 | [Github](https://github.com/vimalabs/VIMA) | Local Demo | 
| ![Star](https://img.shields.io/github/stars/MineDojo/MineDojo.svg?style=social&label=Star) <br> [**MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge**](https://arxiv.org/pdf/2206.08853.pdf) <br> | NeurIPS | 2022-06-17 | [Github](https://github.com/MineDojo/MineDojo) | - |


## Milestone Papers

|  Date  |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2017-06 |     Transformers     |      Google      | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS   |
| 2018-06 |       GPT 1.0       |      OpenAI      | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                 |            |
| 2018-10 |         BERT         |      Google      | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)                                                              |    NAACL    |
| 2019-02 |       GPT 2.0       |      OpenAI      | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)              |            |
| 2019-09 |     Megatron-LM     |      NVIDIA      | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)                                                          |            |
| 2019-10 |          T5          |      Google      | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)                                                           |    JMLR    |
| 2019-10 |          ZeRO          |      Microsoft      | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)                                                           |    SC    |
| 2020-01 |     Scaling Law     |      OpenAI      | [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)                                                                                                        |            |
| 2020-05 |       GPT 3.0       |      OpenAI      | [Language models are few-shot learners](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)                                                             |   NeurIPS   |
| 2021-01 | Switch Transformers |      Google      | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)                                                   |    JMLR    |
| 2021-08 |        Codex        |      OpenAI      | [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)                                                                                               |            |
| 2021-08 |  Foundation Models  |     Stanford     | [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf)                                                                                            |            |
| 2021-09 |         FLAN         |      Google      | [Finetuned Language Models are Zero-Shot Learners](https://openreview.net/forum?id=gEZrGCozdqR)                                                                                        |    ICLR    |
| 2021-10 |         T0         |      HuggingFace et al.      | [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)                                                                                        |    ICLR    |
| 2021-12 |         GLaM         |      Google      | [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905.pdf)                                                                             |    ICML    |
| 2021-12 |        WebGPT        |      OpenAI      | [WebGPT: Improving the Factual Accuracy of Language Models through Web Browsing](https://openai.com/blog/webgpt/)                                                                      |            |
| 2021-12 |        Retro        |     DeepMind     | [Improving language models by retrieving from trillions of tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)             |    ICML    |
| 2021-12 |        Gopher        |     DeepMind     | [Scaling Language Models: Methods, Analysis &amp; Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf)                                                                 |            |
| 2022-01 |         COT         |      Google      | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)                                                                          |   NeurIPS   |
| 2022-01 |        LaMDA        |      Google      | [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)                                                                                                 |            |
| 2022-01 |        Minerva      |      Google      | [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)                                                                                                 |   NeurIPS         |
| 2022-01 | Megatron-Turing NLG | Microsoft&NVIDIA | [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/pdf/2201.11990.pdf)                                        |            |
| 2022-03 |     InstructGPT     |      OpenAI      | [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)                                                                            |            |
| 2022-04 |         PaLM         |      Google      | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf)                                                                                                  |            |
| 2022-04 |      Chinchilla      |     DeepMind     | [An empirical analysis of compute-optimal large language model training](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training) |   NeurIPS   |
| 2022-05 |         OPT         |       Meta       | [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068.pdf)                                                                                              |            |
| 2022-05 |         UL2         |       Google       | [Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131v1)                                                                                              |            |
| 2022-06 |  Emergent Abilities  |      Google      | [Emergent Abilities of Large Language Models](https://openreview.net/pdf?id=yzkSU5zdwD)                                                                                                |    TMLR    |
| 2022-06 |      BIG-bench      |      Google      | [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://github.com/google/BIG-bench)                                                    |            |
| 2022-06 |        METALM        |    Microsoft    | [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)                                                                                                 |            |
| 2022-09 |       Sparrow       |     DeepMind     | [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/pdf/2209.14375.pdf)                                                                           |            |
| 2022-10 |       Flan-T5/PaLM       |      Google      | [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)                                                                                                  |            |
| 2022-10 |       GLM-130B       |     Tsinghua     | [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/pdf/2210.02414.pdf)                                                                                                  |    ICLR    |
| 2022-11 |         HELM         |     Stanford     | [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110.pdf)                                                                                                         |            |
| 2022-11 |        BLOOM        |    BigScience    | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/pdf/2211.05100.pdf)                                                                                |            |
| 2022-11 |      Galactica      |       Meta       | [Galactica: A Large Language Model for Science](https://arxiv.org/pdf/2211.09085.pdf)                                                                                                  |            |
| 2022-12 | OPT-IML |  Meta | [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/pdf/2212.12017)  |            |
| 2023-01 | Flan 2022 Collection |      Google      | [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf)                                                               |            |
| 2023-02 | LLaMA|Meta|[LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)||
| 2023-02 | Kosmos-1|Microsoft|[Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/abs/2302.14045)||
| 2023-03 | PaLM-E | Google | [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io)||
| 2023-03 | GPT 4 | OpenAI | [GPT-4 Technical Report](https://openai.com/research/gpt-4)||
| 2023-04 | Pythia | EleutherAI et al. | [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)|ICML|
| 2023-05 | Dromedary | CMU et al. | [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision](https://arxiv.org/abs/2305.03047)||
| 2023-05 | PaLM 2 | Google | [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf)||
| 2023-05 | RWKV | Bo Peng | [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) ||
| 2024-02 | Microsoft | |[The-Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764.pdf)||


## Others
|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/1zhou-Wang/MemVR.svg?style=social&label=Star) <br> [**Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models**](https://arxiv.org/pdf/2410.03577.pdf) <br> | arXiv | 2024-10-07 | [Github](https://github.com/1zhou-Wang/MemVR) | - |
| ![Star](https://img.shields.io/github/stars/jonathan-roberts1/charting-new-territories.svg?style=social&label=Star) <br> [**Charting New Territories: Exploring the Geographic and Geospatial Capabilities of Multimodal LLMs**](https://arxiv.org/pdf/2311.14656.pdf) <br> | arXiv | 2023-11-24 | [Github](https://github.com/jonathan-roberts1/charting-new-territories) | - |
| [**Can Large Pre-trained Models Help Vision Models on Perception Tasks?**](https://arxiv.org/pdf/2306.00693.pdf) | arXiv | 2023-06-01 | [Coming soon]() | - | 
| ![Star](https://img.shields.io/github/stars/yuhangzang/ContextDET.svg?style=social&label=Star) <br> [**Contextual Object Detection with Multimodal Large Language Models**](https://arxiv.org/pdf/2305.18279.pdf) <br> | arXiv | 2023-05-29 | [Github](https://github.com/yuhangzang/ContextDET) | [Demo](https://huggingface.co/spaces/yuhangzang/ContextDet-Demo) |
| ![Star](https://img.shields.io/github/stars/kohjingyu/gill.svg?style=social&label=Star) <br> [**Generating Images with Multimodal Language Models**](https://arxiv.org/pdf/2305.17216.pdf) <br> | arXiv | 2023-05-26 | [Github](https://github.com/kohjingyu/gill) | - |
| ![Star](https://img.shields.io/github/stars/yunqing-me/AttackVLM.svg?style=social&label=Star) <br> [**On Evaluating Adversarial Robustness of Large Vision-Language Models**](https://arxiv.org/pdf/2305.16934.pdf) <br> | arXiv | 2023-05-26 | [Github](https://github.com/yunqing-me/AttackVLM) | - | 
| ![Star](https://img.shields.io/github/stars/RUCAIBox/POPE.svg?style=social&label=Star) <br> [**Evaluating Object Hallucination in Large Vision-Language Models**](https://arxiv.org/pdf/2305.10355.pdf) <br> | arXiv | 2023-05-17 | [Github](https://github.com/RUCAIBox/POPE) | - |
| ![Star](https://img.shields.io/github/stars/kohjingyu/fromage.svg?style=social&label=Star) <br> [**Grounding Language Models to Images for Multimodal Inputs and Outputs**](https://arxiv.org/pdf/2301.13823.pdf) <br> | ICML | 2023-01-31 | [Github](https://github.com/kohjingyu/fromage) | [Demo](https://huggingface.co/spaces/jykoh/fromage) |

---

## Awesome Datasets

## Datasets of Pre-Training for Alignment
| Name | Paper | Type | Modalities |
|:-----|:-----:|:----:|:----------:|
| **MS-COCO** | [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf) | Caption | Image-Text |
| **SBU Captions** | [Im2Text: Describing Images Using 1 Million Captioned Photographs](https://proceedings.neurips.cc/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) | Caption | Image-Text |
| **Conceptual Captions** | [Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning](https://aclanthology.org/P18-1238.pdf) | Caption | Image-Text |
| **LAION-400M** | [LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs](https://arxiv.org/pdf/2111.02114.pdf) | Caption | Image-Text |
| **VG Captions** |[Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations](https://link.springer.com/content/pdf/10.1007/s11263-016-0981-7.pdf) | Caption | Image-Text |
| **Flickr30k** | [Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models](https://openaccess.thecvf.com/content_iccv_2015/papers/Plummer_Flickr30k_Entities_Collecting_ICCV_2015_paper.pdf) | Caption | Image-Text |
| **AI-Caps** | [AI Challenger : A Large-scale Dataset for Going Deeper in Image Understanding](https://arxiv.org/pdf/1711.06475.pdf) | Caption | Image-Text | 
| **Wukong Captions** | [Wukong: A 100 Million Large-scale Chinese Cross-modal Pre-training Benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/file/a90b9a09a6ee43d6631cf42e225d73b4-Paper-Datasets_and_Benchmarks.pdf) | Caption | Image-Text |
| **Youku-mPLUG** | [Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks](https://arxiv.org/pdf/2306.04362.pdf) | Caption | Video-Text |
| **MSR-VTT** | [MSR-VTT: A Large Video Description Dataset for Bridging Video and Language](https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf) | Caption | Video-Text |
| **Webvid10M** | [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/pdf/2104.00650.pdf) | Caption | Video-Text |
| **WavCaps** | [WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research](https://arxiv.org/pdf/2303.17395.pdf) | Caption | Audio-Text |
| **AISHELL-1** | [AISHELL-1: An open-source Mandarin speech corpus and a speech recognition baseline](https://arxiv.org/pdf/1709.05522.pdf) | ASR | Audio-Text |
| **AISHELL-2** | [AISHELL-2: Transforming Mandarin ASR Research Into Industrial Scale](https://arxiv.org/pdf/1808.10583.pdf) | ASR | Audio-Text |
| **VSDial-CN** | [X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages](https://arxiv.org/pdf/2305.04160.pdf) | ASR | Image-Audio-Text |


## Tutorials about LLM
- [Andrej Karpathy] State of GPT [video](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)
- [Hyung Won Chung] Instruction finetuning and RLHF lecture [Youtube](https://www.youtube.com/watch?v=zjrM-MW-0y0)
- [Jason Wei] Scaling, emergence, and reasoning in large language models [Slides](https://docs.google.com/presentation/d/1EUV7W7X_w0BDrscDhPg7lMGzJCkeaPkGCJ3bN8dluXc/edit?pli=1&resourcekey=0-7Nz5A7y8JozyVrnDtcEKJA#slide=id.g16197112905_0_0)
- [Susan Zhang] Open Pretrained Transformers [Youtube](https://www.youtube.com/watch?v=p9IxoSkvZ-M&t=4s)
- [Ameet Deshpande] How Does ChatGPT Work? [Slides](https://docs.google.com/presentation/d/1TTyePrw-p_xxUbi3rbmBI3QQpSsTI1btaQuAUvvNc8w/edit#slide=id.g206fa25c94c_0_24)
- [Yao Fu] 预训练，指令微调，对齐，专业化：论大语言模型能力的来源 [Bilibili](https://www.bilibili.com/video/BV1Qs4y1h7pn/?spm_id_from=333.337.search-card.all.click&vd_source=1e55c5426b48b37e901ff0f78992e33f)
- [Hung-yi Lee] ChatGPT 原理剖析 [Youtube](https://www.youtube.com/watch?v=yiY4nPOzJEg&list=RDCMUC2ggjtuuWvxrHHHiaDH1dlQ&index=2)
- [Jay Mody] GPT in 60 Lines of NumPy [Link](https://jaykmody.com/blog/gpt-from-scratch/)
- [ICML 2022] Welcome to the &#34;Big Model&#34; Era: Techniques and Systems to Train and Serve Bigger Models [Link](https://icml.cc/virtual/2022/tutorial/18440)
- [NeurIPS 2022] Foundational Robustness of Foundation Models [Link](https://nips.cc/virtual/2022/tutorial/55796)
- [Andrej Karpathy] Let's build GPT: from scratch, in code, spelled out. [Video](https://www.youtube.com/watch?v=kCc8FmEb1nY)|[Code](https://github.com/karpathy/ng-video-lecture)
- [DAIR.AI] Prompt Engineering Guide [Link](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [邱锡鹏] 大型语言模型的能力分析与应用 [Slides](resources/大型语言模型的能力分析与应用%20-%2030min.pdf) | [Video](https://www.bilibili.com/video/BV1Xb411X7c3/?buvid=XY2DA82257CC34DECD40B00CAE8AFB7F3B43C&is_story_h5=false&mid=dM1oVipECo22eTYTWkJVVg%3D%3D&p=1&plat_id=116&share_from=ugc&share_medium=android&share_plat=android&share_session_id=c42b6c60-9d22-4c75-90b8-48828e1168af&share_source=WEIXIN&share_tag=s_i&timestamp=1676812375&unique_k=meHB9Xg&up_id=487788801&vd_source=1e55c5426b48b37e901ff0f78992e33f)
- [Philipp Schmid] Fine-tune FLAN-T5 XL/XXL using DeepSpeed & Hugging Face Transformers [Link](https://www.philschmid.de/fine-tune-flan-t5-deepspeed)
- [HuggingFace] Illustrating Reinforcement Learning from Human Feedback (RLHF) [Link](https://huggingface.co/blog/rlhf)
- [HuggingFace] What Makes a Dialog Agent Useful? [Link](https://huggingface.co/blog/dialog-agents)
- [张俊林]通向AGI之路：大型语言模型(LLM)技术精要 [Link](https://zhuanlan.zhihu.com/p/597586623)
- [大师兄]ChatGPT/InstructGPT详解 [Link](https://zhuanlan.zhihu.com/p/590311003)
- [HeptaAI]ChatGPT内核：InstructGPT，基于反馈指令的PPO强化学习 [Link](https://zhuanlan.zhihu.com/p/589747432)
- [Yao Fu] How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources [Link](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
- [Stephen Wolfram] What Is ChatGPT Doing … and Why Does It Work? [Link](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)
- [Jingfeng Yang] Why did all of the public reproduction of GPT-3 fail? [Link](https://jingfengyang.github.io/gpt)
- [Hung-yi Lee] ChatGPT (可能)是怎麼煉成的 - GPT 社會化的過程 [Video](https://www.youtube.com/watch?v=e0aKI2GGZNg)
- [Keyvan Kambakhsh] Pure Rust implementation of a minimal Generative Pretrained Transformer [code](https://github.com/keyvank/femtoGPT)

## Open Source LLM
- [LLaMA2](https://about.fb.com/news/2023/07/llama-2/) - A revolutionary version of llama , 70 - 13 - 7 -billion-parameter large language model. [LLaMA2](https://github.com/facebookresearch/llama) [HF - TheBloke/Llama-2-13B-GPTQ](https://huggingface.co/TheBloke/Llama-2-13B-GPTQ)
- [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) - A foundational, 65-billion-parameter large language model. [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama)
  - [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) - A model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations. [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp) [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)
  - [Flan-Alpaca](https://github.com/declare-lab/flan-alpaca) - Instruction Tuning from Humans and Machines.
  - [Baize](https://github.com/project-baize/baize-chatbot) - Baize is an open-source chat model trained with [LoRA](https://github.com/microsoft/LoRA). It uses 100k dialogs generated by letting ChatGPT chat with itself. 
  - [Cabrita](https://github.com/22-hours/cabrita) - A portuguese finetuned instruction LLaMA.
  - [Vicuna](https://github.com/lm-sys/FastChat) - An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. 
  - [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) - An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. 
  - [Llama-X](https://github.com/AetherCortex/Llama-X) - Open Academic Research on Improving LLaMA to SOTA LLM.
  - [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna) - A Chinese Instruction-following LLaMA-based Model.
  - [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) - 4 bits quantization of [LLaMA](https://arxiv.org/abs/2302.13971) using [GPTQ](https://arxiv.org/abs/2210.17323).
  - [GPT4All](https://github.com/nomic-ai/gpt4all) - Demo, data, and code to train open-source assistant-style large language model based on GPT-J and LLaMa.
  - [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/) - A Dialogue Model for Academic Research
  - [BELLE](https://github.com/LianjiaTech/BELLE) - Be Everyone's Large Language model Engine
  - [StackLLaMA](https://huggingface.co/blog/stackllama) - A hands-on guide to train LLaMA with RLHF.
  - [RedPajama](https://github.com/togethercomputer/RedPajama-Data) -  An Open Source Recipe to Reproduce LLaMA training dataset.
  - [Chimera](https://github.com/FreedomIntelligence/LLMZoo) - Latin Phoenix.
  - [WizardLM|WizardCoder](https://github.com/nlpxucan/WizardLM)  - Family of instruction-following LLMs powered by Evol-Instruct: WizardLM, WizardCoder.
  - [CaMA](https://github.com/zjunlp/CaMA) - a Chinese-English Bilingual LLaMA Model.
  - [Orca](https://aka.ms/orca-lm) - Microsoft's finetuned LLaMA model that reportedly matches GPT3.5, finetuned against 5M of data, ChatGPT, and GPT4
  - [BayLing](https://github.com/ictnlp/BayLing) - an English/Chinese LLM equipped with advanced language alignment, showing superior capability in English/Chinese generation, instruction following and multi-turn interaction.
  - [UltraLM](https://github.com/thunlp/UltraChat) - Large-scale, Informative, and Diverse Multi-round Chat Models.
  - [Guanaco](https://github.com/artidoro/qlora) - QLoRA tuned LLaMA
- [BLOOM](https://huggingface.co/bigscience/bloom) - BigScience Large Open-science Open-access Multilingual Language Model [BLOOM-LoRA](https://github.com/linhduongtuan/BLOOM-LORA)
  - [BLOOMZ&mT0](https://huggingface.co/bigscience/bloomz) - a family of models capable of following human instructions in dozens of languages zero-shot.
  - [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)
- [T5](https://arxiv.org/abs/1910.10683) - Text-to-Text Transfer Transformer 
  - [T0](https://arxiv.org/abs/2110.08207) - Multitask Prompted Training Enables Zero-Shot Task Generalization
- [OPT](https://arxiv.org/abs/2205.01068) - Open Pre-trained Transformer Language Models.
- [UL2](https://arxiv.org/abs/2205.05131v1) - a unified framework for pretraining models that are universally effective across datasets and setups. 
- [GLM](https://github.com/THUDM/GLM)- GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.
  - [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) - ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数.
  - [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) - An Open Bilingual Chat LLM | 开源双语对话语言模型
- [RWKV](https://github.com/BlinkDL/RWKV-LM) - Parallelizable RNN with Transformer-level LLM Performance.
  - [ChatRWKV](https://github.com/BlinkDL/ChatRWKV) - ChatRWKV is like ChatGPT but powered by my RWKV (100% RNN) language model.
- [StableLM](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) - Stability AI Language Models.
- [YaLM](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6) - a GPT-like neural network for generating and processing text. It can be used freely by developers and researchers from all over the world.
- [GPT-Neo](https://github.com/EleutherAI/gpt-neo) - An implementation of model & data parallel [GPT3](https://arxiv.org/abs/2005.14165)-like models using the [mesh-tensorflow](https://github.com/tensorflow/mesh) library.
- [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b) - A 6 billion parameter, autoregressive text generation model trained on [The Pile](https://pile.eleuther.ai/).
  - [Dolly](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) - a cheap-to-build LLM that exhibits a surprising degree of the instruction following capabilities exhibited by ChatGPT.
- [Pythia](https://github.com/EleutherAI/pythia) - Interpreting Autoregressive Transformers Across Time and Scale
- [Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) - the first open source, instruction-following LLM, fine-tuned on a human-generated instruction dataset licensed for research and commercial use.
- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) - an open-source reproduction of DeepMind's Flamingo model.
- [Cerebras-GPT](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/) - A Family of Open, Compute-efficient, Large Language Models.
- [GALACTICA](https://github.com/paperswithcode/galai/blob/main/docs/model_card.md) - The GALACTICA models are trained on a large-scale scientific corpus.
  - [GALPACA](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-30b) - GALACTICA 30B fine-tuned on the Alpaca dataset.
- [Palmyra](https://huggingface.co/Writer/palmyra-base) - Palmyra Base was primarily pre-trained with English text.
- [Camel](https://huggingface.co/Writer/camel-5b-hf) - a state-of-the-art instruction-following large language model designed to deliver exceptional performance and versatility.
- [h2oGPT](https://github.com/h2oai/h2ogpt)
- [PanGu-α](https://openi.org.cn/pangu/) - PanGu-α is a 200B parameter autoregressive pretrained Chinese language model develped by Huawei Noah's Ark Lab, MindSpore Team and Peng Cheng Laboratory.
- [MOSS](https://github.com/OpenLMLab/MOSS) - MOSS是一个支持中英双语和多种插件的开源对话语言模型.
- [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) - a project meant to give everyone access to a great chat based large language model.
  - [HuggingChat](https://huggingface.co/chat/) - Powered by Open Assistant's latest model – the best open source chat model right now and @huggingface Inference API.
- [StarCoder](https://huggingface.co/blog/starcoder) - Hugging Face LLM for Code
- [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) - Open LLM for commercial use by MosaicML
- [Falcon](https://falconllm.tii.ae) - Falcon LLM is a foundational large language model (LLM) with 40 billion parameters trained on one trillion tokens. TII has now released Falcon LLM – a 40B model.
- [XGen](https://github.com/salesforce/xgen) - Salesforce open-source LLMs with 8k sequence length.
- [baichuan-7B](https://github.com/baichuan-inc/baichuan-7B) - baichuan-7B 是由百川智能开发的一个开源可商用的大规模预训练语言模型.
- [Aquila](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila) - 悟道·天鹰语言大模型是首个具备中英双语知识、支持商用许可协议、国内数据合规需求的开源语言大模型。
  
## LLM Training Frameworks
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) - DeepSpeed version of NVIDIA's Megatron-LM that adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and others. 
- [FairScale](https://fairscale.readthedocs.io/en/latest/what_is_fairscale.html) - FairScale is a PyTorch extension library for high performance and large scale training.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Ongoing research training transformer models at scale.
- [Colossal-AI](https://github.com/hpcaitech/ColossalAI) - Making large AI models cheaper, faster, and more accessible.
- [BMTrain](https://github.com/OpenBMB/BMTrain) - Efficient Training for Big Models.
- [Mesh Tensorflow](https://github.com/tensorflow/mesh) - Mesh TensorFlow: Model Parallelism Made Easier.
- [maxtext](https://github.com/google/maxtext) - A simple, performant and scalable Jax LLM!
- [Alpa](https://alpa.ai/index.html) - Alpa is a system for training and serving large-scale neural networks.
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) - An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library.

## Tools for deploying LLM

- [FastChat](https://github.com/lm-sys/FastChat) - A distributed multi-model LLM serving system with web UI and OpenAI-compatible RESTful APIs.
- [SkyPilot](https://github.com/skypilot-org/skypilot) - Run LLMs and batch jobs on any cloud. Get maximum cost savings, highest GPU availability, and managed execution -- all with a simple interface.
- [vLLM](https://github.com/vllm-project/vllm) - A high-throughput and memory-efficient inference and serving engine for LLMs
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - A Rust, Python and gRPC server for text generation inference. Used in production at [HuggingFace](https://huggingface.co/) to power LLMs api-inference widgets.
- [Haystack](https://haystack.deepset.ai/) - an open-source NLP framework that allows you to use LLMs and transformer-based models from Hugging Face, OpenAI and Cohere to interact with your own data. 
-  [Sidekick](https://github.com/ai-sidekick/sidekick) - Data integration platform for LLMs. 
- [LangChain](https://github.com/hwchase17/langchain) -  Building applications with LLMs through composability 
-  [wechat-chatgpt](https://github.com/fuergaosi233/wechat-chatgpt) - Use ChatGPT On Wechat via wechaty
- [promptfoo](https://github.com/typpo/promptfoo) - Test your prompts. Evaluate and compare LLM outputs, catch regressions, and improve prompt quality.
- [Agenta](https://github.com/agenta-ai/agenta) -  Easily build, version, evaluate and deploy your LLM-powered apps.
- [Embedchain](https://github.com/embedchain/embedchain) - Framework to create ChatGPT like bots over your dataset.

## Courses about LLM

- [DeepLearning.AI] ChatGPT Prompt Engineering for Developers [Homepage](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [Princeton] Understanding Large Language Models [Homepage](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
- [OpenBMB] 大模型公开课 [主页](https://www.openbmb.org/community/course)
- [Stanford] CS224N-Lecture 11: Prompting, Instruction Finetuning, and RLHF [Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture11-prompting-rlhf.pdf)
- [Stanford] CS324-Large Language Models [Homepage](https://stanford-cs324.github.io/winter2022/)
- [Stanford] CS25-Transformers United V2 [Homepage](https://web.stanford.edu/class/cs25/)
- [Stanford Webinar] GPT-3 & Beyond [Video](https://www.youtube.com/watch?v=-lnHHWRCDGk)
- [李沐] InstructGPT论文精读 [Bilibili](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.337.search-card.all.click&vd_source=1e55c5426b48b37e901ff0f78992e33f) [Youtube](https://www.youtube.com/watch?v=zfIGAwD1jOQ)
- [陳縕儂] OpenAI InstructGPT 從人類回饋中學習 ChatGPT 的前身 [Youtube](https://www.youtube.com/watch?v=ORHv8yKAV2Q)
- [李沐] HELM全面语言模型评测 [Bilibili](https://www.bilibili.com/video/BV1z24y1B7uX/?spm_id_from=333.337.search-card.all.click&vd_source=1e55c5426b48b37e901ff0f78992e33f)
- [李沐] GPT，GPT-2，GPT-3 论文精读 [Bilibili](https://www.bilibili.com/video/BV1AF411b7xQ/?spm_id_from=333.788&vd_source=1e55c5426b48b37e901ff0f78992e33f) [Youtube](https://www.youtube.com/watch?v=t70Bl3w7bxY&list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I&index=18)
- [Aston Zhang] Chain of Thought论文 [Bilibili](https://www.bilibili.com/video/BV1t8411e7Ug/?spm_id_from=333.788&vd_source=1e55c5426b48b37e901ff0f78992e33f) [Youtube](https://www.youtube.com/watch?v=H4J59iG3t5o&list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I&index=29)
- [MIT] Introduction to Data-Centric AI [Homepage](https://dcai.csail.mit.edu)

## Datasets of Multimodal Instruction Tuning
| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **Video-ChatGPT** | [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/pdf/2306.05424.pdf) | [Link](https://github.com/mbzuai-oryx/Video-ChatGPT#video-instruction-dataset-open_file_folder) | 100K high-quality video instruction dataset | 
| **MIMIC-IT** | [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](https://arxiv.org/pdf/2306.05425.pdf) | [Coming soon](https://github.com/Luodian/Otter) | Multimodal in-context instruction tuning |
| **M<sup>3</sup>IT** | [M<sup>3</sup>IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning](https://arxiv.org/pdf/2306.04387.pdf) | [Link](https://huggingface.co/datasets/MMInstruction/M3IT) | Large-scale, broad-coverage multimodal instruction tuning dataset | 
| **LLaVA-Med** | [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/pdf/2306.00890.pdf) | [Coming soon](https://github.com/microsoft/LLaVA-Med#llava-med-dataset) | A large-scale, broad-coverage biomedical instruction-following dataset |
| **GPT4Tools** | [GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction](https://arxiv.org/pdf/2305.18752.pdf) | [Link](https://github.com/StevenGrove/GPT4Tools#dataset) | Tool-related instruction datasets |
| **MULTIS** | [ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst](https://arxiv.org/pdf/2305.16103.pdf) | [Coming soon](https://iva-chatbridge.github.io/) | Multimodal instruction tuning dataset covering 16 multimodal tasks |
| **DetGPT** | [DetGPT: Detect What You Need via Reasoning](https://arxiv.org/pdf/2305.14167.pdf) | [Link](https://github.com/OptimalScale/DetGPT/tree/main/dataset) |  Instruction-tuning dataset with 5000 images and around 30000 query-answer pairs|
| **PMC-VQA** | [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://arxiv.org/pdf/2305.10415.pdf) | [Coming soon](https://xiaoman-zhang.github.io/PMC-VQA/) | Large-scale medical visual question-answering dataset |
| **VideoChat** | [VideoChat: Chat-Centric Video Understanding](https://arxiv.org/pdf/2305.06355.pdf) | [Link](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) | Video-centric multimodal instruction dataset |
| **X-LLM** | [X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages](https://arxiv.org/pdf/2305.04160.pdf) | [Link](https://github.com/phellonchen/X-LLM) | Chinese multimodal instruction dataset |
| **OwlEval** | [mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality](https://arxiv.org/pdf/2304.14178.pdf) | [Link](https://github.com/X-PLUG/mPLUG-Owl/tree/main/OwlEval) | Dataset for evaluation on multiple capabilities |
| **cc-sbu-align** | [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/pdf/2304.10592.pdf) | [Link](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) | Multimodal aligned dataset for improving model's usability and generation's fluency |
| **LLaVA-Instruct-150K** | [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485.pdf) | [Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Multimodal instruction-following data generated by GPT|
| **MultiInstruct** | [MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning](https://arxiv.org/pdf/2212.10773.pdf) | - | The first multimodal instruction tuning benchmark dataset |

## Other useful resources
- [Mistral](https://mistral.ai/) - Mistral-7B-v0.1 is a small, yet powerful model adaptable to many use-cases including code and 8k sequence length. Apache 2.0 licence.
- [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) - a high-quality sparse mixture of experts model (SMoE) with open weights.
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) - an experimental open-source application showcasing the capabilities of the GPT-4 language model. 
- [OpenAGI](https://github.com/agiresearch/OpenAGI) - When LLM Meets Domain Experts.
- [HuggingGPT](https://github.com/microsoft/JARVIS) - Solving AI Tasks with ChatGPT and its Friends in HuggingFace.
- [EasyEdit](https://github.com/zjunlp/EasyEdit) - An easy-to-use framework to edit large language models.
- [chatgpt-shroud](https://github.com/guyShilo/chatgpt-shroud) - A Chrome extension for OpenAI's ChatGPT, enhancing user privacy by enabling easy hiding and unhiding of chat history. Ideal for privacy during screen shares.
- [Arize-Phoenix](https://phoenix.arize.com/) - Open-source tool for ML observability that runs in your notebook environment. Monitor and fine tune LLM, CV and Tabular Models.
- [Emergent Mind](https://www.emergentmind.com) - The latest AI news, curated & explained by GPT-4.
- [ShareGPT](https://sharegpt.com) - Share your wildest ChatGPT conversations with one click.
- [Major LLMs + Data Availability](https://docs.google.com/spreadsheets/d/1bmpDdLZxvTCleLGVPgzoMTQ0iDP2-7v7QziPrzPdHyM/edit#gid=0)
- [500+ Best AI Tools](https://vaulted-polonium-23c.notion.site/500-Best-AI-Tools-e954b36bf688404ababf74a13f98d126)
- [Cohere Summarize Beta](https://txt.cohere.ai/summarize-beta/) - Introducing Cohere Summarize Beta: A New Endpoint for Text Summarization
- [chatgpt-wrapper](https://github.com/mmabrouk/chatgpt-wrapper) - ChatGPT Wrapper is an open-source unofficial Python API and CLI that lets you interact with ChatGPT.
- [Open-evals](https://github.com/open-evals/evals) - A framework extend openai's [Evals](https://github.com/openai/evals) for different language model.
- [Cursor](https://www.cursor.so) - Write, edit, and chat about your code with a powerful AI.


## Prompting libraries & tools

- [YiVal](https://github.com/YiVal/YiVal) — Evaluate and Evolve: YiVal is an open-source GenAI-Ops tool for tuning and evaluating prompts, configurations, and model parameters using customizable datasets, evaluation methods, and improvement strategies.
- [Guidance](https://github.com/microsoft/guidance) — A handy looking Python library from Microsoft that uses Handlebars templating to interleave generation, prompting, and logical control.
- [LangChain](https://github.com/hwchase17/langchain) — A popular Python/JavaScript library for chaining sequences of language model prompts.
- [FLAML (A Fast Library for Automated Machine Learning & Tuning)](https://microsoft.github.io/FLAML/docs/Getting-Started/): A Python library for automating selection of models, hyperparameters, and other tunable choices.
- [Chainlit](https://docs.chainlit.io/overview) — A Python library for making chatbot interfaces.
- [Guardrails.ai](https://www.guardrailsai.com/docs/) — A Python library for validating outputs and retrying failures. Still in alpha, so expect sharp edges and bugs.
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) — A Python/C#/Java library from Microsoft that supports prompt templating, function chaining, vectorized memory, and intelligent planning.
- [Prompttools](https://github.com/hegelai/prompttools) — Open-source Python tools for testing and evaluating models, vector DBs, and prompts.
- [Outlines](https://github.com/normal-computing/outlines) — A Python library that provides a domain-specific language to simplify prompting and constrain generation.
- [Promptify](https://github.com/promptslab/Promptify) — A small Python library for using language models to perform NLP tasks.
- [Scale Spellbook](https://scale.com/spellbook) — A paid product for building, comparing, and shipping language model apps.
- [PromptPerfect](https://promptperfect.jina.ai/prompts) — A paid product for testing and improving prompts.
- [Weights & Biases](https://wandb.ai/site/solutions/llmops) — A paid product for tracking model training and prompt engineering experiments.
- [OpenAI Evals](https://github.com/openai/evals) — An open-source library for evaluating task performance of language models and prompts.
- [LlamaIndex](https://github.com/jerryjliu/llama_index) — A Python library for augmenting LLM apps with data.
- [Arthur Shield](https://www.arthur.ai/get-started) — A paid product for detecting toxicity, hallucination, prompt injection, etc.
- [LMQL](https://lmql.ai) — A programming language for LLM interaction with support for typed prompting, control flow, constraints, and tools.
- [ModelFusion](https://github.com/lgrammel/modelfusion) - A TypeScript library for building apps with LLMs and other ML models (speech-to-text, text-to-speech, image generation).
- [Flappy](https://github.com/pleisto/flappy) — Production-Ready LLM Agent SDK for Every Developer.
- [GPTRouter](https://gpt-router.writesonic.com/) - GPTRouter is an open source LLM API Gateway that offers a universal API for 30+ LLMs, vision, and image models, with smart fallbacks based on uptime and latency, automatic retries, and streaming. Stay operational even when OpenAI is down

## Datasets of In-Context Learning

| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **MIMIC-IT** | [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](https://arxiv.org/pdf/2306.05425.pdf) | [Coming soon](https://github.com/Luodian/Otter) | Multimodal in-context instruction dataset|

## Datasets of Multimodal Chain-of-Thought

| Name | Paper | Link | Notes |
|:-----|:-----:|:----:|:-----:|
| **EgoCOT** | [EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought](https://arxiv.org/pdf/2305.15021.pdf) | [Coming soon](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) | Large-scale embodied planning dataset |
| **VIP** | [Let’s Think Frame by Frame: Evaluating Video Chain of Thought with Video Infilling and Prediction](https://arxiv.org/pdf/2305.13903.pdf) | [Coming soon]() | An inference-time dataset that can be used to evaluate VideoCOT |
| **ScienceQA** | [Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf) | [Link](https://github.com/lupantech/ScienceQA#ghost-download-the-dataset) | Large-scale multi-choice dataset, featuring multimodal science questions and diverse domains | 

## Practical Guide for Data

## Pretraining data
- **RedPajama**, 2023. [Repo](https://github.com/togethercomputer/RedPajama-Data)
- **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**, Arxiv 2020. [Paper](https://arxiv.org/abs/2101.00027)
- **How does the pre-training objective affect what large language models learn about linguistic properties?**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.16/)
- **Scaling laws for neural language models**, 2020. [Paper](https://arxiv.org/abs/2001.08361)
- **Data-centric artificial intelligence: A survey**, 2023. [Paper](https://arxiv.org/abs/2303.10158)
- **How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources**, 2022. [Blog](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
- 
### Finetuning data
- **Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach**, EMNLP 2019. [Paper](https://arxiv.org/abs/1909.00161)
- **Language Models are Few-Shot Learners**, NIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
- **Does Synthetic Data Generation of LLMs Help Clinical Text Mining?** Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04360)

### Test data/user data
- **Shortcut learning of large language models in natural language understanding: A survey**, Arxiv 2023. [Paper](https://arxiv.org/abs/2208.11857)
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective** Arxiv, 2023. [Paper](https://arxiv.org/abs/2302.12095)
- **SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems** Arxiv 2019. [Paper](https://arxiv.org/abs/1905.00537)





## Practical Guide for NLP Tasks
We build a decision flow for choosing LLMs or fine-tuned models~\protect\footnotemark for user's NLP applications. The decision flow helps users assess whether their downstream NLP applications at hand meet specific conditions and, based on that evaluation, determine whether LLMs or fine-tuned models are the most suitable choice for their applications.
<p align="center">
<img width="500" src="./imgs/decision.png"/>  
</p>

### Traditional NLU tasks

- **A benchmark for toxic comment classification on civil comments dataset** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.11125)
- **Is chatgpt a general-purpose natural language processing task solver?** Arxiv 2023[Paper](https://arxiv.org/abs/2302.06476)
- **Benchmarking large language models for news summarization** Arxiv 2022 [Paper](https://arxiv.org/abs/2301.13848)
### Generation tasks
- **News summarization and evaluation in the era of gpt-3** Arxiv 2022 [Paper](https://arxiv.org/abs/2209.12356)
- **Is chatgpt a good translator? yes with gpt-4 as the engine** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.08745)
- **Multilingual machine translation systems from Microsoft for WMT21 shared task**, WMT2021 [Paper](https://aclanthology.org/2021.wmt-1.54/)
- **Can ChatGPT understand too? a comparative study on chatgpt and fine-tuned bert**, Arxiv 2023, [Paper](https://arxiv.org/pdf/2302.10198.pdf)




### Knowledge-intensive tasks
- **Measuring massive multitask language understanding**, ICLR 2021 [Paper](https://arxiv.org/abs/2009.03300)
- **Beyond the imitation game: Quantifying and extrapolating the capabilities of language models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2206.04615)
- **Inverse scaling prize**, 2022 [Link](https://github.com/inverse-scaling/prize)
- **Atlas: Few-shot Learning with Retrieval Augmented Language Models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2208.03299)
- **Large Language Models Encode Clinical Knowledge**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.13138)


### Abilities with Scaling

- **Training Compute-Optimal Large Language Models**, NeurIPS 2022 [Paper](https://openreview.net/pdf?id=iBBcRUlOAPR)
- **Scaling Laws for Neural Language Models**, Arxiv 2020 [Paper](https://arxiv.org/abs/2001.08361)
- **Solving math word problems with process- and outcome-based feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.14275)
- **Chain of thought prompting elicits reasoning in large language models**, NeurIPS 2022 [Paper](https://arxiv.org/abs/2201.11903)
- **Emergent abilities of large language models**, TMLR 2022 [Paper](https://arxiv.org/abs/2206.07682)
- **Inverse scaling can become U-shaped**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.02011)
- **Towards Reasoning in Large Language Models: A Survey**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10403)


### Specific tasks
- **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**, Arixv 2022 [Paper](https://arxiv.org/abs/2208.10442)
- **PaLI: A Jointly-Scaled Multilingual Language-Image Model**, Arxiv 2022 [Paper](https://arxiv.org/abs/2209.06794)
- **AugGPT: Leveraging ChatGPT for Text Data Augmentation**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.13007)
- **Is gpt-3 a good data annotator?**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10450)
- **Want To Reduce Labeling Cost? GPT-3 Can Help**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.354/)
- **GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.192/)
- **LLM for Patient-Trial Matching: Privacy-Aware Data Augmentation Towards Better Performance and Generalizability**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16756)
- **ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.15056)
- **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16634)
- **GPTScore: Evaluate as You Desire**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.04166)
- **Large Language Models Are State-of-the-Art Evaluators of Translation Quality**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.14520)
- **Is ChatGPT a Good NLG Evaluator? A Preliminary Study**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04048)
- **GPT4GEO: How a Language Model Sees the World's Geography**, NeurIPSW 2023 [Paper](https://arxiv.org/abs/2306.00020), [Code](https://github.com/jonathan-roberts1/GPT4GEO)

### Real-World ''Tasks''
- **Sparks of Artificial General Intelligence: Early experiments with GPT-4**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.12712)

### Efficiency
1. Cost
- **Openai’s gpt-3 language model: A technical overview**, 2020. [Blog Post](https://lambdalabs.com/blog/demystifying-gpt-3)
- **Measuring the carbon intensity of ai in cloud instances**, FaccT 2022. [Paper](https://dl.acm.org/doi/abs/10.1145/3531146.3533234)
- **In AI, is bigger always better?**, Nature Article 2023. [Article](https://www.nature.com/articles/d41586-023-00641-w)
- **Language Models are Few-Shot Learners**, NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- **Pricing**, OpenAI. [Blog Post](https://openai.com/pricing)
2. Latency
- HELM: **Holistic evaluation of language models**, Arxiv 2022. [Paper](https://arxiv.org/abs/2211.09110)
3. Parameter-Efficient Fine-Tuning
- **LoRA: Low-Rank Adaptation of Large Language Models**, Arxiv 2021. [Paper](https://arxiv.org/abs/2106.09685)
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**, ACL 2021. [Paper](https://aclanthology.org/2021.acl-long.353/)
- **P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.8/)
- **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**, Arxiv 2022. [Paper](https://arxiv.org/abs/2110.07602)
4. Pretraining System
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**, Arxiv 2019. [Paper](https://arxiv.org/abs/1910.02054)
- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**, Arxiv 2019. [Paper](https://arxiv.org/abs/1910.02054)
- **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**, Arxiv 2021. [Paper](https://arxiv.org/abs/2104.04473)
- **Reducing Activation Recomputation in Large Transformer Models**, Arxiv 2021. [Paper](https://arxiv.org/abs/2104.04473)

  
## RLHFdataset

- [HH-RLHF](https://github.com/anthropics/hh-rlhf)
  - Ben Mann, Deep Ganguli
  - Keyword: Human preference dataset, Red teaming data, machine-written
  - Task: Open-source dataset for human preference data about helpfulness and harmlessness
- [Stanford Human Preferences Dataset(SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)
  - Ethayarajh, Kawin and Zhang, Heidi and Wang, Yizhong and Jurafsky, Dan
  - Keyword: Naturally occurring and human-written dataset,18 different subject areas
  - Task: Intended to be used for training RLHF reward models
- [PromptSource](https://github.com/bigscience-workshop/promptsource)
  - Stephen H. Bach, Victor Sanh, Zheng-Xin Yong et al.
  - Keyword: Prompted English datasets,  Mapping a data example into natural language
  - Task:  Toolkit for creating, Sharing and using natural language prompts
- [Structured Knowledge Grounding(SKG) Resources Collections](https://unifiedskg.com/)
  - Tianbao Xie, Chen Henry Wu, Peng Shi et al.
  - Keyword: Structured Knowledge Grounding
  - Task:  Collection of datasets are related to structured knowledge grounding
- [The Flan Collection](https://github.com/google-research/FLAN/tree/main/flan/v2)
  - Longpre Shayne, Hou Le, Vu Tu et al.
  - Task: Collection compiles datasets from Flan 2021, P3, Super-Natural Instructions 
- [rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)
  - Yiting Xie
  - Keyword: Machine-written dataset
- [webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)
  - OpenAI
  - Keyword: Human-written dataset, Long form question answering 
  - Task:  Train a long form question answering model to align with human preferences
- [summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)
  - OpenAI
  - Keyword: Human-written dataset, summarization
  - Task:  Train a summarization model to align with human preferences
- [Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise)
  - Dahoas
  - Keyword: Human-written dataset, synthetic dataset
- [Stable Alignment - Alignment Learning in Social Games](https://github.com/agi-templar/Stable-Alignment)
  - Ruibo Liu, Ruixin (Ray) Yang, Qiang Peng
  - Keyword: Interaction data used for alignment training, Run in Sandbox
  - Task: Train on the recorded interaction data in simulated social games
- [LIMA](https://huggingface.co/datasets/GAIR/lima)
  - Meta AI
  - Keyword: without any RLHF, few carefully curated prompts and responses
  - Task: Dataset used for training the LIMA model

  - Zhenyu Hou, Pengfan Du, Yilin Niu, Zhengxiao Du, Aohan Zeng, Xiao Liu, Minlie Huang, Hongning Wang, Jie Tang, Yuxiao Dong
- [Search, Verify and Feedback: Towards Next Generation Post-training Paradigm of Foundation Models via Verifier Engineering](https://arxiv.org/abs/2411.11504)
  - Xinyan Guan, Yanjiang Liu, Xinyu Lu, Boxi Cao, Ben He, Xianpei Han, Le Sun, Jie Lou, Bowen Yu, Yaojie Lu, Hongyu Lin
- [Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective](https://arxiv.org/abs/2412.14135)
  - Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Bo Wang, Shimin Li, Yunhua Zhou, Qipeng Guo, Xuanjing Huang, Xipeng Qiu
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
  - Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman
  - https://github.com/ezelikman/quiet-star
- [Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision](https://arxiv.org/abs/2411.16579)
  - Zhiheng Xi, Dingwen Yang, Jixuan Huang, Jiafu Tang, Guanyu Li, Yiwen Ding, Wei He, Boyang Hong, Shihan Do, Wenyu Zhan, Xiao Wang, Rui Zheng, Tao Ji, Xiaowei Shi, Yitao Zhai, Rongxiang Weng, Jingang Wang, Xunliang Cai, Tao Gui, Zuxuan Wu, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Yu-Gang Jiang
  - https://mathcritique.github.io/
- [On Designing Effective RL Reward at Training Time for LLM Reasoning](https://arxiv.org/abs/2410.15115)
  - Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, Yi Wu
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240)
  - Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal
- [Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](https://arxiv.org/abs/2410.08146)
  - Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, Aviral Kumar
- [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)
  - Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, Abhinav Rastogi
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)
  - Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, Deli Chen, Y.Wu, Zhifang Sui
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733)
  - Evan Wang, Federico Cassano, Catherine Wu, Yunfeng Bai, Will Song, Vaskar Nath, Ziwen Han, Sean Hendryx, Summer Yue, Hugh Zhang
- [PROCESSBENCH: Identifying Process Errors in Mathematical Reasoning](https://arxiv.org/pdf/2412.06559)
  - Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin
- [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762)
  - Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, Chenglin Wu
- [Interpretable Contrastive Monte Carlo Tree Search Reasoning](https://arxiv.org/abs/2410.01707)
  - Zitian Gao, Boye Niu, Xuzheng He, Haotian Xu, Hongzhang Liu, Aiwei Liu, Xuming Hu, Lijie Wen
- [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199)
  - Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, Rafael Rafailov
- [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)
  - Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, James Zou
- [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271)
  - Zhiyuan Hu, Chumin Liu, Xidong Feng, Yilun Zhao, See-Kiong Ng, Anh Tuan Luu, Junxian He, Pang Wei Koh, Bryan Hooi
- [Advancing LLM Reasoning Generalists with Preference Trees](https://arxiv.org/abs/2404.02078)
  - Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan et al.
- [Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing](https://arxiv.org/abs/2404.12253)
  - Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu.
- [AlphaMath Almost Zero: Process Supervision Without Process](https://arxiv.org/abs/2405.03553)
  - Guoxin Chen, Minpeng Liao, Chengxi Li, Kai Fan.
- [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)
  - Dan Zhang, Sining Zhoubian, Yisong Yue, Yuxiao Dong, and Jie Tang.
- [Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search](https://arxiv.org/abs/2412.18319)
  - Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, Dacheng Tao
- [Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](https://arxiv.org/abs/2411.14432)
  - Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, Ziwei Liu
- [MindStar: Enhancing Math Reasoning in Pre-trained LLMs at Inference Time](https://arxiv.org/abs/2405.16265)
  - Jikun Kang, Xin Zhe Li, Xi Chen, Amirreza Kazemi, Qianyi Sun, Boxing Chen, Dong Li, Xu He, Quan He, Feng Wen, Jianye Hao, Jun Yao.
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)
  - Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P. Lillicrap, Kenji Kawaguchi, Michael Shieh.
- [When is Tree Search Useful for LLM Planning? It Depends on the Discriminator](https://arxiv.org/abs/2402.10890)
  - Ziru Chen, Michael White, Raymond Mooney, Ali Payani, Yu Su, Huan Sun
- [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875)
  - Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma.
- [To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning](https://arxiv.org/abs/2409.12183)
  - Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, Greg Durrett
- [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837)
  - Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, Sebastian Riedel
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/pdf/2402.10200)
  - Xuezhi Wang, Denny Zhou
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
  - Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang
- [Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2406.09136)
  - Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, Min Lin
- [ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967)
  - Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, Hang Li
- [VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment](https://arxiv.org/abs/2410.01679)
  - Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, Nicolas Le Roux
- [Stream of Search (SoS): Learning to Search in Language](https://arxiv.org/abs/2404.03683)
  - Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, Noah D. Goodman
- [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://arxiv.org/abs/2410.05229)
  - Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar
- [Evaluation of OpenAI o1: Opportunities and Challenges of AGI](https://arxiv.org/abs/2409.18486)
  - Tianyang Zhong, Zhengliang Liu, Yi Pan, Yutong Zhang, Yifan Zhou, Shizhe Liang, Zihao Wu, Yanjun Lyu, Peng Shu, Xiaowei Yu, Chao Cao, Hanqi Jiang, Hanxu Chen, Yiwei Li, Junhao Chen, etc.
- [Evaluating LLMs at Detecting Errors in LLM Responses](https://arxiv.org/abs/2404.03602)
  - Ryo Kamoi, Sarkar Snigdha Sarathi Das, Renze Lou, Jihyun Janice Ahn, Yilun Zhao, Xiaoxin Lu, Nan Zhang, Yusen Zhang, Ranran Haoran Zhang, Sujeeth Reddy Vummanthala, Salika Dave, Shaobo Qin, Arman Cohan, Wenpeng Yin, Rui Zhang
- [On The Planning Abilities of OpenAI's o1 Models: Feasibility, Optimality, and Generalizability](https://arxiv.org/abs/2409.19924)
  - Kevin Wang, Junbo Li, Neel P. Bhatt, Yihan Xi, Qiang Liu, Ufuk Topcu, Zhangyang Wang
- [Not All LLM Reasoners Are Created Equal](https://arxiv.org/abs/2410.01748)
  - Arian Hosseini, Alessandro Sordoni, Daniel Toyama, Aaron Courville, Rishabh Agarwal
- [LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench](https://arxiv.org/abs/2409.13373)
  - Karthik Valmeekam, Kaya Stechly, Subbarao Kambhampati
- [A Comparative Study on Reasoning Patterns of OpenAI's o1 Model](https://arxiv.org/abs/2410.13639)
  - Siwei Wu, Zhongyuan Peng, Xinrun Du, Tuney Zheng, Minghao Liu, Jialong Wu, Jiachen Ma, Yizhi Li, Jian Yang, Wangchunshu Zhou, Qunshu Lin, Junbo Zhao, Zhaoxiang Zhang, Wenhao Huang, Ge Zhang, Chenghua Lin, J.H. Liu
- [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630)
  - Tianhao Wu, Janice Lan, Weizhe Yuan, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar
- [Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning Through Trap Problems](https://arxiv.org/pdf/2405.06680)
  - Jun Zhao, Jingqi Tong, Yurong Mou, Ming Zhang, Qi Zhang, Xuanjing Huang
- [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457)
  - Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal
- [CPL: Critical Plan Step Learning Boosts LLM Generalization in Reasoning Tasks](https://arxiv.org/abs/2409.08642)
  - Tianlong Wang, Junzhe Chen, Xueting Han, Jing Bai
- [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089)
  - Tianhao Wu, Janice Lan, Weizhe Yuan, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar
- [Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283)
  - Chaojie Wang, Yanchen Deng, Zhiyi Lyu, Liang Zeng, Jujie He, Shuicheng Yan, Bo An

### 2023
- [Training Chain-of-Thought via Latent-Variable Inference](https://arxiv.org/pdf/2312.02179)
  - Du Phan, Matthew D. Hoffman, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous
- [Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training](https://arxiv.org/abs/2309.17179)
  - Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, Jun Wang
- [OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning](https://arxiv.org/pdf/2311.09724)
  - Fei Yu, Anningzhe Gao, Benyou Wang
- [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)
  - Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu
- [Don’t throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding](https://arxiv.org/abs/2309.15028)
  - Liu, Jiacheng, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh Hajishirzi, and Asli Celikyilmaz.
- [Certified reasoning with language models](https://arxiv.org/pdf/2306.04031)
  - Gabriel Poesia, Kanishk Gandhi, Eric Zelikman, Noah D. Goodman
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)
  - Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou

### 2022
- [Chain of Thought Imitation with Procedure Cloning](https://arxiv.org/abs/2205.10816)
  - Mengjiao Yang, Dale Schuurmans, Pieter Abbeel, Ofir Nachum.
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
  - Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
- [Solving math word problems with processand outcome-based feedback](https://arxiv.org/abs/2211.14275)
  - Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, Irina Higgins

### 2021
- [Scaling Scaling Laws with Board Games](http://arxiv.org/abs/2104.03113)
  - Andy L. Jones.
- [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/pdf/2112.00114)
  - Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, Augustus Odena

### 2017
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815v1)
  - David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis.

# Test data/user data
- **Shortcut learning of large language models in natural language understanding: A survey**, Arxiv 2023. [Paper](https://arxiv.org/abs/2208.11857)
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective** Arxiv, 2023. [Paper](https://arxiv.org/abs/2302.12095)
- **SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems** Arxiv 2019. [Paper](https://arxiv.org/abs/1905.00537)

7. EEC: "**Examining Gender and Race Bias in Two Hundred Sentiment Analysis Systems**". ![](https://img.shields.io/badge/Dataset-blue)

    *Kiritchenko Svetlana et al.* NAACL HLT 2018. [[Paper](https://arxiv.org/pdf/1805.04508)] [[Source](https://saifmohammad.com/WebPages/Biases-SA.html)] 

8. WikiGenderBias: "**Towards Understanding Gender Bias in Relation Extraction**". ![](https://img.shields.io/badge/Dataset-blue)

    *Gaut Andrew et al.* ACL 2020. [[Paper](https://arxiv.org/pdf/1911.03642)] [[GitHub](https://github.com/AndrewJGaut/Towards-Understanding-Gender-Bias-in-Neural-Relation-Extraction)] 

9. "**Measuring and Mitigating Unintended Bias in Text Classification**". ![](https://img.shields.io/badge/Dataset-blue)

    *Lucas Dixon et al.* AAAI 2018. [[Paper](https://dl.acm.org/doi/10.1145/3278721.3278729)] [[GitHub](https://github.com/conversationai/unintended-ml-bias-analysis)] 

10. "**Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification**". ![](https://img.shields.io/badge/Dataset-blue)

    *Daniel Borkan et al.* WWW 2019. [[Paper](https://arxiv.org/abs/1903.04561)] 

11. "**Social Bias Frames: Reasoning about Social and Power Implications of Language**". ![](https://img.shields.io/badge/Dataset-blue)

    *Sap Maarten  et al.* ACL 2020. [[Paper](https://arxiv.org/abs/1911.03891)] [[Source](https://huggingface.co/datasets/social_bias_frames)] 

12. "**Finding Microaggressions in the Wild: A Case for Locating Elusive  Phenomena in Social Media Posts**". ![](https://img.shields.io/badge/Dataset-blue)

    *Breitfeller Luke et al.* EMNLP-IJCNLP 2019. [[Paper](https://aclanthology.org/D19-1176.pdf)] 

13. Latent Hatred: "**Latent Hatred: A Benchmark for Understanding Implicit Hate Speech**". ![](https://img.shields.io/badge/Dataset-blue)

    *Mai ElSherief et al.* EMNLP 2021. [[Paper](https://arxiv.org/pdf/2109.05322)] [[GitHub](https://github.com/gt-salt/implicit-hate)] 

14. DynaHate: "**Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection**". ![](https://img.shields.io/badge/Dataset-blue)

    *Vidgen Bertie et al.* ACL/IJCNLP 2021. [[Paper](https://arxiv.org/pdf/2012.15761)] [[GitHub](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)] 

15. TOXIGEN: "**ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection**". ![](https://img.shields.io/badge/Dataset-blue)

    *Thomas Hartvigsen et al.* ACL 2022. [[Paper](https://arxiv.org/pdf/2203.09509)] [[GitHub](https://github.com/microsoft/TOXIGEN)] [[Source](https://huggingface.co/datasets/skg/toxigen-data)] 

16. CDail-Bias: "**Towards Identifying Social Bias in Dialog Systems: Frame, Datasets, and Benchmarks**". ![](https://img.shields.io/badge/Dataset-blue)

    *Jingyan Zhou et al.* EMNLP 2022. [[Paper](https://arxiv.org/pdf/2202.08011)] [[GitHub](https://github.com/para-zhou/CDial-Bias)] 

17. CORGI-PM: "**CORGI-PM: A Chinese Corpus For Gender Bias Probing and Mitigation**". ![](https://img.shields.io/badge/Dataset-blue)

    *Ge Zhang et al.* arXiv  2023. [[Paper](https://arxiv.org/pdf/2301.00395)] [[GitHub](https://github.com/yizhilll/CORGI-PM)] 

18. HateCheck: "**HateCheck: Functional Tests for Hate Speech Detection Models**". ![](https://img.shields.io/badge/Dataset-blue)

    *Paul Röttger et al.* ACL/IJCNLP 2021. [[Paper](https://arxiv.org/abs/2012.15606)] [[GitHub](https://github.com/paul-rottger/hatecheck-data)] 

19. StereoSet: "**StereoSet: Measuring stereotypical bias in pretrained language models**". ![](https://img.shields.io/badge/Dataset-blue)

    *Moin Nadeem et al.* ACL/IJCNLP 2021. [[Paper](https://arxiv.org/abs/2004.09456)] [[GitHub](https://github.com/moinnadeem/StereoSet)] [[Source](https://huggingface.co/datasets/stereoset)] 

20. CrowS-Pairs: "**CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models**". ![](https://img.shields.io/badge/Dataset-blue)

    *Nikita Nangia et al.* EMNLP 2020. [[Paper](https://arxiv.org/abs/2010.00133)] [[GitHub](https://github.com/nyu-mll/crows-pairs)] [[Source](https://huggingface.co/datasets/crows_pairs)] 

21. "**Does gender matter? towards fairness in dialogue systems**". ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Evaluation%20Method-orange)

    *Haochen Liu et al.* COLING 2020.  [[Paper](https://arxiv.org/abs/1910.10486)] [[GitHub](https://github.com/zgahhblhc/DialogueFairness)] 

22. BOLD: "**BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation**". ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Evaluation%20Method-orange)

    *Jwala Dhamala et al.* FAccT 2021. [[Paper](https://arxiv.org/abs/2101.11718)] [[GitHub](https://github.com/amazon-science/bold)] [[Source](https://huggingface.co/datasets/AlexaAI/bold)] 

23. HolisticBias: "**“I’m sorry to hear that”: Finding New Biases in Language Models with a Holistic Descriptor Dataset**". ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Evaluation%20Method-orange)

    *Eric Michael Smith et al.* EMNLP 2022. [[Paper](https://arxiv.org/abs/2205.09209)] [[GitHub](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias)] 

24. Multilingual Holistic Bias: "**Multilingual Holistic Bias: Extending Descriptors and Patterns to Unveil Demographic Biases in Languages at Scale**". ![](https://img.shields.io/badge/Dataset-blue)

    *Eric Michael Smith et al.* arXiv  2023. [[Paper](https://arxiv.org/abs/2305.13198)] 

25. Unqover: "**UNQOVERing Stereotyping Biases via Underspecified Questions**". ![](https://img.shields.io/badge/Dataset-blue)

    *Tao Li et al.* EMNLP 2020. [[Paper](https://arxiv.org/abs/2010.02428)] [[GitHub](https://github.com/allenai/unqover)] 

26. BBQ: **"BBQ: A Hand-Built Bias Benchmark for Question Answering**".  ![](https://img.shields.io/badge/Dataset-blue)

    *Alicia Parrish et al.* ACL 2022. [[Paper](https://arxiv.org/abs/2110.08193)] [[GitHub](https://github.com/nyu-mll/bbq)]

27. CBBQ: "**CBBQ: A Chinese Bias Benchmark Dataset Curated with Human-AI Collaboration for Large Language Models**". ![](https://img.shields.io/badge/Dataset-blue)

    *Yufei Huang et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2306.16244)] [[GitHub](https://github.com/yfhuangxxxx/cbbq)] 

28. "**Gender Bias in Multilingual Embeddings and Cross-Lingual Transfer**". ![](https://img.shields.io/badge/Dataset-blue)

    *Jieyu Zhao et al.* ACL 2020. [[Paper](https://arxiv.org/abs/2005.00699)] [[GitHub](https://github.com/MSR-LIT/MultilingualBias)] 

29. FairLex: "**FairLex: A Multilingual Benchmark for Evaluating Fairness in Legal Text Processing**". ![](https://img.shields.io/badge/Dataset-blue)

    *Ilias Chalkidis et al.* ACL 2022. [[Paper](https://arxiv.org/abs/2203.07228)] [[GitHub](https://github.com/coastalcph/fairlex)] 

30. "**Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification**". ![](https://img.shields.io/badge/Evaluation%20Method-orange)

    *Daniel Borkan et al.* WWW 2019. [[Paper](https://arxiv.org/abs/1903.04561)] 

31. "**On measuring and mitigating biased inferences of word embeddings**". ![](https://img.shields.io/badge/Evaluation%20Method-orange)

   *Sunipa Dev et al.* AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6267/6123)] 

32. "**An Empirical Study of Metrics to Measure Representational Harms in Pre-Trained Language Models**". ![](https://img.shields.io/badge/Evaluation%20Method-orange)

    *Saghar Hosseini et al.* TrustNLP 2023. [[Paper](https://arxiv.org/pdf/2301.09211)] [[GitHub](https://github.com/microsoft/SafeNLP)] 

33. "**Revealing Persona Biases in Dialogue Systems**". ![](https://img.shields.io/badge/Evaluation%20Method-orange)

    *Emily Sheng et al.* arXiv  2021. [[Paper](https://arxiv.org/abs/2104.08728)] [[GitHub](https://github.com/ewsheng/persona-biases)] 

34. "**On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?** ". ![](https://img.shields.io/badge/Rearch-green)

    *Emily M. Bender et al.* FAccT 2021. [[Paper](https://doi.org/10.1145/3442188.3445922)] 

35. "**A Survey on Hate Speech Detection using Natural Language Processing.**" ![](https://img.shields.io/badge/Rearch-green)

    *Anna Schmidt et al.* SocialNLP 2017. [[Paper](https://aclanthology.org/W17-1101/)] 

36. "**Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity**". ![](https://img.shields.io/badge/Evaluation%20Method-orange)![](https://img.shields.io/badge/Rearch-green)

    *Terry Yue Zhuo et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12867)]


    *Yushi Bai et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2308.14508.pdf)] [[GitHub](https://github.com/THUDM/LongBench)] 

#### Benchmarks for Knowledge and Reasoning

1. MMLU: **"Measuring Massive Multitask Language Understanding"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Dan Hendrycks et al.* ICLR 2021. [[Paper](http://arxiv.org/abs/1506.06724v1)] [[GitHub](https://github.com/hendrycks/test)] 

2. MMCU: **"Measuring Massive Multitask Chinese Understanding"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Hui Zeng et al.* arXiv 2023. [[Paper](https://arxiv.org/ftp/arxiv/papers/2304/2304.12986.pdf)] [[GitHub](https://github.com/Felixgithub2017/MMCU)] 

3. C-Eval: **"C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models"**. ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

    *Yuzhen Huang et al. arXiv* 2023. [[Paper](https://arxiv.org/pdf/2305.08322.pdf)] [[Source](https://cevalbenchmark.com/)] 

4. M3KE: **"M3KE: A Massive Multi-Level Multi-Subject Knowledge Evaluation Benchmark for Chinese Large Language Models"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Chuang Liu et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2305.10263.pdf)] [[GitHub](https://github.com/tjunlp-lab/M3KE)] 

5. CMMLU: **"CMMLU: Measuring massive multitask language understanding in Chinese"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Haonan Li et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.09212.pdf)] [[GitHub](https://github.com/haonan-li/CMMLU)] 

6. AGIEval: **"AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Wanjun Zhong et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2304.06364.pdf)] [[GitHub](https://github.com/ruixiangcui/AGIEval)] 

7. M3Exam: **"M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Wenxuan Zhang et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.05179.pdf)] [[GitHub](https://github.com/DAMO-NLP-SG/M3Exam)] 

8. LucyEval: **"Evaluating the Generation Capabilities of Large Chinese Language Models"**. ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

    *Hui Zeng* *et al.* arXiv 2023. [[Paper](https://arxiv.org/ftp/arxiv/papers/2308/2308.04823.pdf)] [[Source](http://cgeval.besteasy.com/)] [[GitHub](https://github.com/Felixgithub2017/CG-Eval)] 

#### Benchmark for Holistic Evaluation

1. Big-bench: **"Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Dan Hendrycks et al.* ICLR 2021. [[Paper](https://arxiv.org/pdf/2206.04615.pdf)] [[GitHub](https://github.com/google/BIG-bench)] 

2. Evaluation Harness: **"A framework for few-shot language model evaluation"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Leo Gao et al.* arXiv 2023. [[GitHub](https://github.com/EleutherAI/lm-evaluation-harness)] 

3. HELM: **"Holistic Evaluation of Language Models"**. ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

    *Yuzhen Huang et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2211.09110.pdf)] [[Source](https://crfm.stanford.edu/helm/v0.1.0)] [[GitHub](https://github.com/stanford-crfm/helm)] 

4. OpenAI Evals [[GitHub](https://github.com/openai/evals)] ![](https://img.shields.io/badge/Dataset-blue)

5. GPT-Fathom: **"GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond"**. ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Rearch-green)

    *Shen Zheng and Yuyu Zhang et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2309.16583.pdf)] [[GitHub](https://github.com/GPT-Fathom/GPT-Fathom)] 

6. **"INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models"**. ![](https://img.shields.io/badge/Dataset-blue)

    *Yew Ken Chia et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.04757.pdf)] [[Source](https://huggingface.co/datasets/declare-lab/InstructEvalImpact)] [[GitHub](https://github.com/declare-lab/instruct-eval)] 

7. Huggingface Open LLM Leaderboard [[Source](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)] ![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

8. Chatbot Arena: **"Judging LLM-as-a-judge with MT-Bench and Chatbot Arena"**. ![](https://img.shields.io/badge/Evaluation%20Platform-yellow)![](https://img.shields.io/badge/Evaluation%20Method-orange)

    *Lianmin Zheng et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2306.05685.pdf)] [[Source](https://chat.lmsys.org/?arena)] [[GitHub](http://https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)] 

9. FlagEval [[Source](https://flageval.baai.ac.cn/)] [[GitHub](https://github.com/FlagOpen/FlagEval)] ![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

10. OpenCompass: **"Evaluating the Generation Capabilities of Large Chinese Language Models"**. ![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

    *Yuan Liu et al.* arXiv 2023. [[Source](https://opencompass.org.cn/)] [[GitHub](https://github.com/open-compass/opencompass)] 

11. CLEVA: **"CLEVA: Chinese Language Models EVAluation Platform"**. ![](https://img.shields.io/badge/Dataset-blue)![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

     *Yanyang Li et al.* arXiv 2023. [[Paper](https://arxiv.org/pdf/2308.04813.pdf)] [[Source](http://www.lavicleva.com/)] [[GitHub](https://github.com/LaVi-Lab/CLEVA)] 

12. OpenEval [[Source](http://openeval.org.cn/#/)] ![](https://img.shields.io/badge/Evaluation%20Platform-yellow)

## LLM Leaderboards

|              Platform               | Access                                                       | Domain                                                       |
| :---------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|            Chatbot Arena            | [[Source](https://chat.lmsys.org/)]                          | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|                CLEVA                | [[Source](http://www.lavicleva.com/)]                        | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|              FlagEval               | [[Source](https://flageval.baai.ac.cn/)]                     | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|                HELM                 | [[Source](https://crfm.stanford.edu/helm/)]                  | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|  Huggingface Open LLM Leaderboard   | [[Source](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)] | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|            InstructEval             | [[Source](https://declare-lab.net/instruct-eval/)]           | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|              LLMonitor              | [[Source](https://benchmarks.llmonitor.com/)]                | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|             OpenCompass             | [[Source](https://opencompass.org.cn/)]                      | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|             Open Ko-LLM             | [[Source](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)] | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|              SuperCLUE              | [[Source](https://www.superclueai.com/)]                     | Evaluation Organization/ Benchmark for Holistic Evaluation   |
| TheoremOne LLM Benchmarking Metrics | [[Source](https://llm-evals.formula-labs.com/)]              | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|               Toloka                | [[Source](https://toloka.ai/llm-leaderboard/)]               | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|     Open Multilingual LLM Eval      | [[Source](https://huggingface.co/spaces/uonlp/open_multilingual_llm_leaderboard)] | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|              OpenEval               | [[Source](http://openeval.org.cn/#/)]                        | Evaluation Organization/ Benchmark for Holistic Evaluation   |
|                ANGO                 | [[Source](https://huggingface.co/spaces/AngoHF/ANGO-Leaderboard)] | Evaluation Organization/ Benchmarks for Knowledge and Reasoning |
|               C-Eval                | [[Source](https://cevalbenchmark.com/)]                      | Evaluation Organization/ Benchmarks for Knowledge and Reasoning |
|              LucyEval               | [[Source](http://cgeval.besteasy.com/)]                      | Evaluation Organization/ Benchmarks for Knowledge and Reasoning |
|                MMLU                 | [[Source](https://huggingface.co/spaces/CoreyMorris/MMLU-by-task-Leaderboard)] | Evaluation Organization/ Benchmarks for Knowledge and Reasoning |
|             OpenKG LLM              | [[Source](https://huggingface.co/spaces/openkg/llm_leaderboard)] | Evaluation Organization/ Benchmarks for Knowledge and Reasoning |
|             SEED-Bench              | [[Source](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard)] | Evaluation Organization/ Benchmarks for NLU and NLG          |
|              SuperGLUE              | [[Source](https://super.gluebenchmark.com/)]                 | Evaluation Organization/ Benchmarks for NLU and NLG          |
|              Toolbench              | [[Source](https://huggingface.co/spaces/qiantong-xu/toolbench-leaderboard)] | Knowledge and Capability Evaluation/ Tool Learning           |
|      Hallucination Leaderboard      | [[Source](https://github.com/vectara/hallucination-leaderboard)] | Alignment Evaluation/ Truthfulness                           |
|             AlpacaEval              | [[Source](https://tatsu-lab.github.io/alpaca_eval/)]         | Alignment Evaluation/ General Alignment Evaluation           |
|             AgentBench              | [[Source](https://llmbench.ai/agent)]                        | Safety Evaluation/ Evaluating LLMs as Agents                 |
|              InterCode              | [[Source](https://intercode-benchmark.github.io/)]           | Safety Evaluation/ Evaluating LLMs as Agents                 |
|             SafetyBench             | [[Source](https://llmbench.ai/safety)]                       | Safety Evaluation                                            |
|       Nucleotide Transformer        | [[Source](https://huggingface.co/spaces/InstaDeepAI/nucleotide_transformer_benchmark)] | Specialized LLMs Evaluation/ Biology and Medicine            |
|                LAiW                 | [[Source](https://huggingface.co/spaces/daishen/LAiW)]       | Specialized LLMs Evaluation/ Legislation                     |
|     Big Code Models Leaderboard     | [[Source](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)] | Specialized LLMs Evaluation/ Computer Science                |
|  Huggingface LLM Perf Leaderboard   | [[Source](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)] | the Performance of LLMs                                      |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Atomic-man007/Awesome_Multimodel_LLM&type=Date)](https://star-history.com/#Atomic-man007/Awesome_Multimodel_LLM)
