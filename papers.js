/**
 * Papers data for Sang-gil Lee's homepage
 * 
 * To add a new paper:
 * 1. Add an entry to the appropriate array (RESEARCH_PAPERS or PROJECT_PAPERS)
 * 2. Add the paper image to the images/ folder
 * 3. The page will automatically render the new paper
 * 
 * Paper object structure:
 * {
 *   id: string,           // Unique identifier (used for HTML element IDs)
 *   title: string,        // Paper title
 *   authors: string,      // HTML string with author names (use <strong> for self, <a> for links)
 *   venue: string,        // Conference/journal name
 *   year: number,         // Publication year
 *   image: string,        // Image filename in images/ folder
 *   highlighted: boolean, // Whether to highlight with yellow background
 *   links: [              // Array of link objects
 *     { text: string, url: string }
 *   ],
 *   description: string,  // Brief description (can include HTML)
 *   award: string         // Optional: award text (e.g., "Best Paper Award")
 * }
 */

const RESEARCH_PAPERS = [
  // ============================================
  // HIGHLIGHTED PAPERS (where I'm (co-)first author)
  // ============================================
  {
    id: "ualm",
    title: "UALM: Unified Audio Language Model for Understanding, Generation, and Reasoning",
    authors: `
      <a href="https://scholar.google.com/citations?user=KE5I4R0AAAAJ&hl">Jinchuan Tian*</a>,
      <strong>Sang-gil Lee*</strong>,
      <a href="https://cseweb.ucsd.edu/~z4kong/">Zhifeng Kong*</a>,
      <a href="https://sreyan88.github.io/">Sreyan Ghosh</a>,
      <a href="https://goelarushi.github.io">Arushi Goel</a>,
      <a href="https://huckiyang.github.io/">Chao-Han Huck Yang</a>,
      <a href="https://wenliangdai.github.io/">Wenliang Dai</a>,
      <a href="https://zliucr.github.io/">Zihan Liu</a>,
      <a href="https://sites.google.com/site/yhrspace/home">Hanrong Ye</a>,
      <a href="https://sites.google.com/view/shinjiwatanabe">Shinji Watanabe</a>,
      <a href="https://scholar.google.com/citations?user=62ElavIAAAAJ&hl=en">Mohammad Shoeybi</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>,
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://wpingnet.github.io/">Wei Ping</a>
    `,
    venue: "International Conference on Learning Representations (ICLR)",
    award: "Oral",
    year: 2026,
    image: "ualm.png",
    highlighted: true,
    links: [
      { text: "Project Page", url: "https://research.nvidia.com/labs/adlr/UALM/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2510.12000" },
      { text: "Code", url: "https://github.com/NVIDIA/audio-intelligence/tree/main/UALM" }
    ],
    description: "UALM is a unified audio language model that supports audio understanding, audio generation, and multimodal reasoning across speech, sound, and music within a single model."
  },
  {
    id: "etta",
    title: "ETTA: Elucidating the Design Space of Text-to-Audio Models",
    authors: `
      <strong>Sang-gil Lee*</strong>,
      <a href="https://cseweb.ucsd.edu/~z4kong/">Zhifeng Kong*</a>,
      <a href="https://goelarushi.github.io">Arushi Goel</a>,
      <a href="https://scholar.google.com/citations?user=6qGppvkAAAAJ">Sungwon Kim</a>,
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "International Conference on Machine Learning (ICML)",
    year: 2025,
    image: "etta.png",
    highlighted: true,
    links: [
      { text: "Project Page", url: "https://research.nvidia.com/labs/adlr/ETTA/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2412.19351" },
      { text: "Code", url: "https://github.com/NVIDIA/audio-intelligence/tree/main/ETTA" }
    ],
    description: "ETTA is the first text-to-audio model with emergent abilities, capable of synthesizing entirely novel, imaginative sounds beyond the real world by leveraging large-scale synthetic audio captions (AF-Synthetic)."
  },
  {
    id: "bigvgan",
    title: "BigVGAN: A Universal Neural Vocoder with Large-Scale Training",
    authors: `
      <strong>Sang-gil Lee</strong>,
      <a href="https://wpingnet.github.io/">Wei Ping</a>,
      <a href="https://scholar.google.com/citations?user=7BRYaGcAAAAJ">Boris Ginsburg</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ/">Bryan Catanzaro</a>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "International Conference on Learning Representations (ICLR)",
    year: 2023,
    image: "bigvgan.png",
    highlighted: true,
    links: [
      { text: "Project Page", url: "https://research.nvidia.com/labs/adlr/projects/bigvgan/" },
      { text: "Model", url: "https://huggingface.co/collections/nvidia/bigvgan-66959df3d97fd7d98d97dc9a" },
      { text: "arXiv", url: "https://arxiv.org/abs/2206.04658" },
      { text: "Code", url: "https://github.com/NVIDIA/BigVGAN" },
      { text: "Demo", url: "https://bigvgan-demo.github.io/" },
      { text: "BigVGAN-v2 Blog Post", url: "https://developer.nvidia.com/blog/achieving-state-of-the-art-zero-shot-waveform-audio-generation-across-audio-types/" }
    ],
    description: "BigVGAN is a universal audio synthesizer that achieves unprecedented zero-shot performance on various unseen environments using anti-aliased periodic nonlinearity and large-scale training."
  },

  // ============================================
  // REGULAR PAPERS (chronological, newest first)
  // ============================================
  {
    id: "personaplex",
    title: "PersonaPlex: Natural Conversational AI With Any Role and Voice",
    authors: `
      <a href="https://scholar.google.com/citations?user=FtoCDKEAAAAJ&hl=en">Rajarshi Roy</a>,
      <a href="https://scholar.google.com/citations?user=xQAoHP0AAAAJ&hl=en">Jonathan Raiman</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://github.com/tdene">Teodor-Dumitru Ene</a>,
      <a href="https://scholar.google.com/citations?hl=en&user=bxH4S2QAAAAJ">Robert Kirby</a>,
      <a href="https://scholar.google.com/citations?user=6qGppvkAAAAJ">Sungwon Kim</a>,
      <a href="https://jaywalnut310.github.io/">Jaehyeon Kim</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)",
    year: 2026,
    image: "personaplex.jpeg",
    highlighted: false,
    links: [
      { text: "Project Page", url: "https://research.nvidia.com/labs/adlr/personaplex/" },
      { text: "Preprint", url: "https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf" },
      { text: "Model", url: "https://huggingface.co/nvidia/personaplex-7b-v1" },
      { text: "Code", url: "https://github.com/NVIDIA/personaplex" }
    ],
    description: "PersonaPlex is a full-duplex conversational AI model that can listen and speak simultaneously with customizable voice and persona, supporting natural turn-taking, interruptions, and backchannels."
  },
  {
    id: "musicflamingo",
    title: "Music Flamingo: Scaling Music Understanding in Audio Language Models",
    authors: `
      <a href="https://sreyan88.github.io/">Sreyan Ghosh*</a>,
      <a href="https://goelarushi.github.io">Arushi Goel*</a>,
      <a href="https://scholar.google.com/citations?user=QcnwixwAAAAJ&hl=en">Lasha Koroshinadze</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://cseweb.ucsd.edu/~z4kong/">Zhifeng Kong</a>,
      <a href="https://scholar.google.com/citations?user=u2tgePAAAAAJ&hl=en">Joao Felipe Santos</a>,
      <a href="https://www.cs.umd.edu/people/ramanid">Ramani Duraiswami</a>,
      <a href="https://www.cs.umd.edu/people/dmanocha">Dinesh Manocha</a>,
      <a href="https://wpingnet.github.io/">Wei Ping</a>,
      <a href="https://scholar.google.com/citations?user=62ElavIAAAAJ&hl=en">Mohammad Shoeybi</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "International Conference on Learning Representations (ICLR)",
    year: 2026,
    image: "music_flamingo.png",
    highlighted: false,
    links: [
      { text: "Project Page", url: "https://research.nvidia.com/labs/adlr/MF/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2511.10289" }
    ],
    description: "Music Flamingo is an audio language model specialized for music understanding, capable of rich music captioning and QA covering harmony, structure, lyrics, and cultural context."
  },
  {
    id: "af3",
    title: "Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models",
    authors: `
      <a href="https://goelarushi.github.io">Arushi Goel*</a>,
      <a href="https://sreyan88.github.io/">Sreyan Ghosh*</a>,
      <a href="https://jaywalnut310.github.io/">Jaehyeon Kim</a>,
      <a href="https://scholar.google.com/citations?user=jiJ2DcEAAAAJ&hl=en">Sonal Kumar</a>,
      <a href="https://cseweb.ucsd.edu/~z4kong/">Zhifeng Kong</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://huckiyang.github.io/">Chao-Han Huck Yang</a>,
      <a href="https://www.cs.umd.edu/people/ramanid">Ramani Duraiswami</a>,
      <a href="https://www.cs.umd.edu/people/dmanocha">Dinesh Manocha</a>,
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "Conference on Neural Information Processing Systems (NeurIPS)",
    year: 2025,
    image: "audio_flamingo_3.png",
    highlighted: false,
    award: "Spotlight",
    links: [
      { text: "Project Page", url: "https://research.nvidia.com/labs/adlr/AF3/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2507.08128" }
    ],
    description: "Audio Flamingo 3 is a fully open large audio-language model supporting speech, sound, and music understanding with long audio reasoning (up to 10 min), voice-to-voice interaction, and multi-turn chat."
  },
  {
    id: "a2sb",
    title: "A2SB: Audio-to-Audio Schrödinger Bridges",
    authors: `
      <a href="https://cseweb.ucsd.edu/~z4kong/">Zhifeng Kong</a>,
      <a href="https://scholar.google.com/citations?user=4x3DhzAAAAAJ&hl=en">Kevin J Shih</a>,
      <a href="https://weilinie.github.io/">Weili Nie</a>,
      <a href="http://latentspace.cc/">Arash Vahdat</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://scholar.google.com/citations?user=u2tgePAAAAAJ&hl=en">Joao Felipe Santos</a>,
      <a href="https://scholar.google.com/citations?user=ZleK6ccAAAAJ">Ante Jukić</a>,
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "NeurIPS Workshop on AI for Music",
    year: 2025,
    image: "a2sb.png",
    highlighted: false,
    links: [
      { text: "Project Page", url: "https://research.nvidia.com/labs/adlr/A2SB/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2501.11311" },
      { text: "Workshop", url: "https://aiformusicworkshop.github.io" }
    ],
    description: "A2SB is an audio restoration model for high-resolution music at 44.1kHz, capable of bandwidth extension and inpainting. It is end-to-end requiring no vocoder, able to restore hour-long audio inputs."
  },
  {
    id: "uniwav",
    title: "UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation",
    authors: `
      <a href="https://alexander-h-liu.github.io/">Alexander H Liu</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://huckiyang.github.io/">Chao-Han Huck Yang</a>,
      <a href="https://yuangongnd.github.io/">Yuan Gong</a>,
      <a href="https://vllab.ee.ntu.edu.tw/ycwang.html">Yu-Chiang Frank Wang</a>,
      <a href="https://people.csail.mit.edu/jrg/">James R Glass</a>,
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "International Conference on Learning Representations (ICLR)",
    year: 2025,
    image: "uniwav.png",
    highlighted: false,
    links: [
      { text: "Demo Page", url: "https://alexander-h-liu.github.io/uniwav-demo.github.io/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2503.00733" }
    ],
    description: "UniWav is a unified framework combining representation learning and generation for speech, enabling both discriminative and generative tasks within a single pre-trained model."
  },
  {
    id: "fugatto",
    title: "Fugatto 1: Foundational Generative Audio Transformer Opus 1",
    authors: `
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://scholar.google.com/citations?user=sk-qH8wAAAAJ&hl=en">Rohan Badlani</a>,
      <a href="https://cseweb.ucsd.edu/~z4kong/">Zhifeng Kong</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://goelarushi.github.io">Arushi Goel</a>,
      <a href="https://scholar.google.com/citations?user=u2tgePAAAAAJ&hl=en">Joao Felipe Santos</a>,
      <a href="https://scholar.google.com/citations?user=0hr9b1cAAAAJ&hl=en">Aya AlJa'fari</a>,
      <a href="https://scholar.google.com/citations?user=6qGppvkAAAAJ">Sungwon Kim</a>,
      <a href="https://scholar.google.com/citations?user=Sz788IIAAAAJ&hl=en">Shuqi Dai</a>,
      <a href="https://scholar.google.com/citations?user=_C-H8_MAAAAJ&hl=en">Siddharth Gururani</a>,
      <a href="https://alexander-h-liu.github.io/">Alexander H Liu</a>,
      <a href="https://scholar.google.com/citations?user=4x3DhzAAAAAJ&hl=en">Kevin J Shih</a>,
      <a href="https://www.linkedin.com/in/ryan-prenger-18797ba1/">Ryan Prenger</a>,
      <a href="https://wpingnet.github.io/">Wei Ping</a>,
      <a href="https://huckiyang.github.io/">Chao-Han Huck Yang</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "International Conference on Learning Representations (ICLR)",
    year: 2025,
    image: "fugatto.webp",
    highlighted: false,
    links: [
      { text: "Project Page", url: "https://fugatto.github.io/" },
      { text: "OpenReview", url: "https://openreview.net/forum?id=B2Fqu7Y2cd" }
    ],
    description: "Fugatto is a foundational generative audio transformer capable of versatile audio synthesis and transformation following free-form text instructions with emergent compositional abilities."
  },
  {
    id: "a2flow",
    title: "A2-Flow: Alignment-Aware Pre-training for Speech Synthesis with Flow Matching",
    authors: `
      <a href="https://scholar.google.com/citations?user=6qGppvkAAAAJ">Sungwon Kim</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://alexander-h-liu.github.io/">Alexander H Liu</a>,
      <a href="https://scholar.google.com/citations?user=u2tgePAAAAAJ&hl=en">Joao Felipe Santos</a>,
      <a href="https://scholar.google.com/citations?user=dnMm8-EAAAAJ&hl=en">Mikyas Desta</a>,
      <a href="https://www.linkedin.com/in/sudheer-kumar-kovela-b6051618/">Sudheer Kovela</a>,
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "preprint",
    year: 2024,
    image: "a2flow.png",
    highlighted: false,
    links: [
      { text: "OpenReview", url: "https://openreview.net/forum?id=e2p1BWR3vq" },
      { text: "Demo (Magpie TTS)", url: "https://build.nvidia.com/nvidia/magpie-tts-flow" }
    ],
    description: "A2-Flow introduces alignment-aware pre-training for speech synthesis using flow matching. The model is deployed as Magpie TTS Flow on NVIDIA Build."
  },
  {
    id: "lfsc",
    title: "Low Frame-rate Speech Codec: a Codec Designed for Fast High-quality Speech LLM Training and Inference",
    authors: `
      <a href="https://scholar.google.com/citations?user=QvS6LVwAAAAJ">Edresson Casanova</a>,
      <a href="https://www.linkedin.com/in/ryan-langman-49401a4b/">Ryan Langman</a>,
      <a href="https://paarthneekhara.github.io">Paarth Neekhara</a>,
      <a href="https://shehzeen.github.io">Shehzeen Hussain</a>,
      <a href="https://scholar.google.com/citations?user=V28bxDwAAAAJ">Jason Li</a>,
      <a href="https://scholar.google.com/citations?user=HHn14y8AAAAJ">Subhankar Ghosh</a>,
      <a href="https://scholar.google.com/citations?user=ZleK6ccAAAAJ">Ante Jukić</a>,
      <strong>Sang-gil Lee</strong>
    `,
    venue: "IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)",
    year: 2025,
    image: "lfsc.png",
    highlighted: false,
    links: [
      { text: "Project Page", url: "https://edresson.github.io/Low-Frame-rate-Speech-Codec/" },
      { text: "Model", url: "https://huggingface.co/nvidia/low-frame-rate-speech-codec-22khz" },
      { text: "arXiv", url: "https://arxiv.org/abs/2409.12117" },
      { text: "Code", url: "https://github.com/NVIDIA/NeMo" }
    ],
    description: "A neural audio codec that leverages finite scalar quantization and adversarial training with large speech language models to achieve high-quality audio compression with a 1.89 kbps bitrate and 21.5 frames per second."
  },
  {
    id: "synthetic",
    title: "Improving Text-To-Audio Models with Synthetic Captions",
    authors: `
      <a href="https://cseweb.ucsd.edu/~z4kong/">Zhifeng Kong*</a>,
      <strong>Sang-gil Lee*</strong>,
      <a href="https://deepanwayx.github.io">Deepanway Ghosal</a>,
      <a href="https://scholar.google.com.sg/citations?user=jPfEvuQAAAAJ">Navonil Majumder</a>,
      <a href="https://scholar.google.com.sg/citations?user=4q8VxIIAAAAJ">Ambuj Mehrish</a>,
      <a href="https://rafaelvalle.github.io">Rafael Valle</a>,
      <a href="https://soujanyaporia.github.io">Soujanya Poria</a>,
      <a href="https://scholar.google.com/citations?user=UZ6kI2AAAAAJ">Bryan Catanzaro</a>
    `,
    venue: "Interspeech SynData4GenAI",
    year: 2024,
    image: "tangoaf.png",
    highlighted: false,
    links: [
      { text: "Dataset", url: "https://github.com/NVIDIA/audio-flamingo/tree/main/labeling_machine" },
      { text: "Model", url: "https://huggingface.co/declare-lab/tango-af-ac-ft-ac" },
      { text: "arXiv", url: "https://arxiv.org/abs/2406.15487" }
    ],
    description: `<a href="https://github.com/NVIDIA/audio-flamingo/tree/main/labeling_machine">AF-AudioSet</a> is a large-scale audio dataset featuring synthetic captions generated by <a href="https://audioflamingo.github.io/">Audio Flamingo</a>, enabling significant improvements in text-to-audio models.`
  },
  {
    id: "voicetailor",
    title: "VoiceTailor: Lightweight Plug-In Adapter for Diffusion-Based Personalized Text-to-Speech",
    authors: `
      <a href="https://gmltmd789.github.io/">Heeseung Kim</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://scholar.google.com/citations?user=gxNOJPEAAAAJ&hl=en">Jiheum Yeom</a>,
      <a href="https://scholar.google.com/citations?user=DDI2oS8AAAAJ&hl=en">Che Hyun Lee</a>,
      <a href="https://scholar.google.com/citations?user=6qGppvkAAAAJ&hl=en">Sungwon Kim</a>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ&hl=en">Sungroh Yoon</a>
    `,
    venue: "Interspeech",
    year: 2024,
    image: "voicetailor.png",
    highlighted: false,
    links: [
      { text: "Project Page", url: "https://voicetailor.github.io/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2408.14739" }
    ],
    description: "VoiceTailor is a one-shot speaker-adaptive text-to-speech model, which proposes combining low-rank adapters to perform speaker adaptation in a parameter-efficient manner."
  },
  {
    id: "editavideo",
    title: "Edit-A-Video: Single Video Editing with Object-Aware Consistency",
    authors: `
      <a href="https://scholar.google.com/citations?user=M8RX0MEAAAAJ">Chaehun Shin*</a>,
      <a href="https://gmltmd789.github.io">Heeseung Kim*</a>,
      Che Hyun Lee,
      <strong>Sang-gil Lee</strong>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "Asian Conference on Machine Learning (ACML)",
    year: 2023,
    image: "editavideo.gif",
    highlighted: false,
    award: "Best Paper Award",
    links: [
      { text: "Project Page", url: "https://edit-a-video.github.io/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2303.07945" }
    ],
    description: "Edit-A-Video is a diffusion-based one-shot video editing model that solves a background inconsistency problem using a new sparse-causal mask blending method."
  },
  {
    id: "priorgrad",
    title: "PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior",
    authors: `
      <strong>Sang-gil Lee</strong>,
      <a href="https://gmltmd789.github.io">Heeseung Kim</a>,
      <a href="https://scholar.google.com/citations?user=M8RX0MEAAAAJ">Chaehun Shin</a>,
      <a href="https://tan-xu.github.io/">Xu Tan</a>,
      <a href="https://changliu00.github.io/">Chang Liu</a>,
      <a href="https://www.microsoft.com/en-us/research/people/meq/">Qi Meng</a>,
      <a href="https://www.microsoft.com/en-us/research/people/taoqin/">Tao Qin</a>,
      <a href="https://weichen-cas.github.io/">Wei Chen</a>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>,
      <a href="https://www.microsoft.com/en-us/research/people/tyliu/">Tie-Yan Liu</a>
    `,
    venue: "International Conference on Learning Representations (ICLR)",
    year: 2022,
    image: "priorgrad.png",
    highlighted: false,
    links: [
      { text: "Project Page", url: "https://speechresearch.github.io/priorgrad/" },
      { text: "arXiv", url: "https://arxiv.org/abs/2106.06406" },
      { text: "Code", url: "https://github.com/microsoft/NeuralSpeech" },
      { text: "Poster", url: "https://iclr.cc/virtual/2022/poster/6445" }
    ],
    description: "PriorGrad presents an efficient method for constructing a data-dependent non-standard Gaussian prior for training and sampling from diffusion models applied to speech synthesis."
  },
  {
    id: "nanoflow",
    title: "NanoFlow: Scalable Normalizing Flows with Sublinear Parameter Complexity",
    authors: `
      <strong>Sang-gil Lee</strong>,
      <a href="https://scholar.google.com/citations?user=6qGppvkAAAAJ">Sungwon Kim</a>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "Neural Information Processing Systems (NeurIPS)",
    year: 2020,
    image: "nanoflow.png",
    highlighted: false,
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/2006.06280" },
      { text: "Code", url: "https://github.com/L0SG/NanoFlow" },
      { text: "Poster", url: "https://nips.cc/virtual/2020/poster/17696" }
    ],
    description: "NanoFlow uses a single neural network for multiple transformation stages in normalizing flows, which provides an efficient compression for flow-based generative models."
  },
  {
    id: "flowavenet",
    title: "FloWaveNet: A Generative Flow for Raw Audio",
    authors: `
      <a href="https://scholar.google.com/citations?user=6qGppvkAAAAJ">Sungwon Kim</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://scholar.google.com/citations?user=AcVToQUAAAAJ">Jongyoon Song</a>,
      <a href="https://jaywalnut310.github.io/">Jaehyeon Kim</a>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "International Conference on Machine Learning (ICML)",
    year: 2019,
    image: "flowavenet.png",
    highlighted: false,
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/1811.02155" },
      { text: "Code", url: "https://github.com/ksw0306/FloWaveNet" },
      { text: "Demo", url: "https://ksw0306.github.io/flowavenet-demo" },
      { text: "Poster", url: "https://pdfs.semanticscholar.org/9e79/377defb3385ae4dfb5e345c85686e27ca7a5.pdf" }
    ],
    description: "FloWaveNet is one of the first flow-based generative models for fast and parallel synthesis of audio waveforms, enabling a likelihood-based neural vocoder without any auxiliary loss."
  },
  {
    id: "ttsql",
    title: "One-Shot Learning for Text-to-SQL Generation",
    authors: `
      <a href="https://scholar.google.com/citations?user=5Psi6aYAAAAJ">Dongjun Lee</a>,
      <a href="https://www.jaesikyoon.com/">Jaesik Yoon</a>,
      <a href="https://scholar.google.com/citations?user=AcVToQUAAAAJ">Jongyoon Song</a>,
      <strong>Sang-gil Lee</strong>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "arXiv preprint",
    year: 2019,
    image: "ttsql.png",
    highlighted: false,
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/1905.11499" }
    ],
    description: "Template-based one-shot text-to-SQL generative model based on a Candidate Search Network & Pointer Network."
  },
  {
    id: "seqgan",
    title: "Polyphonic Music Generation with Sequence Generative Adversarial Networks",
    authors: `
      <strong>Sang-gil Lee</strong>,
      <a href="https://sites.google.com/view/uiwon-hwang">Uiwon Hwang</a>,
      <a href="https://scholar.google.co.kr/citations?user=dWKk68wAAAAJ">Seonwoo Min</a>,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "arXiv preprint",
    year: 2017,
    image: "seqgan.png",
    highlighted: false,
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/1710.11418" },
      { text: "Code", url: "https://github.com/L0SG/seqgan-music" }
    ],
    description: "This work investigates an efficient musical word representation from polyphonic MIDI data for SeqGAN, simultaneously capturing chords and melodies with dynamic timings."
  },
  {
    id: "snn",
    title: "An Efficient Approach to Boosting Performance of Deep Spiking Network Training",
    authors: `
      Seongsik Park,
      <strong>Sang-gil Lee</strong>,
      Hyunha Nam,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "Neural Information Processing Systems (NIPS) Workshop on Computing with Spikes",
    year: 2016,
    image: "snn.png",
    highlighted: false,
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/1611.02416" }
    ],
    description: "Investigates various initialization and backward control schemes of the membrane potential for training deep spiking networks."
  }
];

const PROJECT_PAPERS = [
  {
    id: "gssdpp",
    title: "Robust End-to-End Focal Liver Lesion Detection Using Unregistered Multiphase Computed Tomography Images",
    authors: `
      <strong>Sang-gil Lee*</strong>,
      <a href="https://sites.google.com/snu.ac.kr/eunjikim">Eunji Kim*</a>,
      <a href="https://scholar.google.co.kr/citations?user=C_GxLiAAAAAJ">Jae Seok Bae*</a>,
      Jung Hoon Kim,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "IEEE Transactions on Emerging Topics in Computational Intelligence (TETCI)",
    year: 2021,
    image: "gssdpp.png",
    highlighted: false,
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/2112.01535" },
      { text: "Code", url: "https://github.com/L0SG/grouped-ssd-pytorch" }
    ],
    description: "GSSD++ provides robustness to unregistered multi-phase CT images for detecting liver lesions using attention-guided multi-phase alignment with deformable convolutions."
  },
  {
    id: "gssd",
    title: "Liver Lesion Detection from Weakly-Labeled Multi-phase CT Volumes with a Grouped Single Shot MultiBox Detector",
    authors: `
      <strong>Sang-gil Lee</strong>,
      <a href="https://scholar.google.co.kr/citations?user=C_GxLiAAAAAJ">Jae Seok Bae</a>,
      Hyunjae Kim,
      Jung Hoon Kim,
      <a href="https://scholar.google.com/citations?user=Bphl_fIAAAAJ">Sungroh Yoon</a>
    `,
    venue: "International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)",
    year: 2018,
    image: "gssd.png",
    highlighted: false,
    links: [
      { text: "arXiv", url: "https://arxiv.org/abs/1807.00436" },
      { text: "Code", url: "https://github.com/L0SG/grouped-ssd-pytorch" }
    ],
    description: "GSSD pioneers a focal liver lesion detection model from multi-phase CT images, which reflects a real-world clinical practice of radiologists."
  }
];

/**
 * Render a single paper as HTML
 */
function renderPaper(paper) {
  const bgColor = paper.highlighted ? ' bgcolor="#ffffd0"' : '';
  const awardHtml = paper.award ? `, <strong>${paper.award}</strong>` : '';
  const linksHtml = paper.links
    .map((link, i) => `<a href="${link.url}">${link.text}</a>`)
    .join(' /\n              ');

  return `
          <tr${bgColor}>
            <td style="padding:20px;width:25%;vertical-align:middle">
              <div class="one">
                <img src='images/${paper.image}' width="160">
              </div>
            </td>
            <td style="padding:20px;width:75%;vertical-align:middle">
              <a href="${paper.links[0]?.url || '#'}">
                <papertitle>${paper.title}</papertitle>
              </a>
              <br>
              ${paper.authors.trim()}
              <br>
              <em>${paper.venue}</em>${awardHtml}, ${paper.year}
              <br> ${linksHtml}
              <p></p>
              <p>${paper.description}</p>
            </td>
          </tr>`;
}

/**
 * Render all papers in a container
 */
function renderPapers(containerId, papers) {
  const container = document.getElementById(containerId);
  if (container) {
    container.innerHTML = papers.map(renderPaper).join('\n');
  }
}

/**
 * Initialize papers on page load
 */
document.addEventListener('DOMContentLoaded', function() {
  renderPapers('research-papers', RESEARCH_PAPERS);
  renderPapers('project-papers', PROJECT_PAPERS);
});
