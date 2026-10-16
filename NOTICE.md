# Third-party notices

CAMP is released under the MIT License ([`LICENSE`](LICENSE)). This repository also includes or adapts the following
third-party code and data.

| Component | Where | Source | License |
|---|---|---|---|
| OpenOOD framework | `main.py`, `openood/`, `configs/`, `scripts/download/download.py` | [Jingkang50/OpenOOD](https://github.com/Jingkang50/OpenOOD) | MIT, Copyright (c) 2021 Jingkang Yang (reproduced in [`LICENSE`](LICENSE)) |
| CLIP stream evaluator, the official AdaNeg and TANL code and their network (`FixedCLIP_NegOODPrompt`), and the WordNet negative-label mining (the implementation of NegLabel's NegMining that NegLabel, TANL, AdaNeg, OODD and CoEvo read) | `openood/evaluators/ood_evaluator_clip_tta.py`, `openood/postprocessors/{adaneg,tanl}/`, `openood/networks/clip_fixed_ood_prompt.py`, `openood/networks/clip_for_wordnet_prepare.py`, `openood/utils/vlm/negative_labels/{wordnet,features}.py` | [PolyU-VCLab/OpenOOD-VLM](https://github.com/PolyU-VCLab/OpenOOD-VLM) | MIT (its `LICENSE` carries the OpenOOD copyright) |
| OpenAI CLIP | `openood/networks/clip/` | [openai/CLIP](https://github.com/openai/CLIP) | MIT, Copyright (c) 2021 OpenAI ([`openood/networks/clip/LICENSE`](openood/networks/clip/LICENSE)) |
| WordNet noun and adjective lists | `data/txtfiles/` | the `txtfiles/` directory of [tmlr-group/NegLabel](https://github.com/tmlr-group/NegLabel), derived from WordNet 3.0 | WordNet 3.0 license, reproduced below (the NegLabel repository itself is Apache-2.0) |
| The official MCM, NegLabel, CSP, OODD and CoEvo code (CoEvo with its network, `FixedCLIP_CoEvo_Neglabel`), and the CSP pool | `openood/postprocessors/{mcm,neglabel,csp,oodd,coevo}/`, `openood/networks/clip_fixed_ood_prompt.py`, `openood/utils/vlm/negative_labels/csp.py` | [deeplearning-wisc/MCM](https://github.com/deeplearning-wisc/MCM), [tmlr-group/NegLabel](https://github.com/tmlr-group/NegLabel), [MengyuanChen21/NeurIPS2024-CSP](https://github.com/MengyuanChen21/NeurIPS2024-CSP), [zxk1212/OODD](https://github.com/zxk1212/OODD), [yuleoliu/CoEvo](https://github.com/yuleoliu/CoEvo) | see each repository |

The WordNet license requires its notice to appear on all copies of the database, so it is reproduced here. It is not
placed inside `data/txtfiles/`, because CSP reads every file in that directory as a word list.

## WordNet 3.0 license

```text
WordNet Release 3.0

This software and database is being provided to you, the LICENSEE, by
Princeton University under the following license.  By obtaining, using
and/or copying this software and database, you agree that you have
read, understood, and will comply with these terms and conditions.:

Permission to use, copy, modify and distribute this software and
database and its documentation for any purpose and without fee or
royalty is hereby granted, provided that you agree to comply with
the following copyright notice and statements, including the disclaimer,
and that the same appear on ALL copies of the software, database and
documentation, including modifications that you make for internal
use or for distribution.

WordNet 3.0 Copyright 2006 by Princeton University.  All rights reserved.

THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON
UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PRINCETON
UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANT-
ABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE
OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT
INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR
OTHER RIGHTS.

The name of Princeton University or Princeton may not be used in
advertising or publicity pertaining to distribution of the software
and/or database.  Title to copyright in this software, database and
any associated documentation shall at all times remain with
Princeton University and LICENSEE agrees to preserve same.
```
