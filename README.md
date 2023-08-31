# Generate then Select: Open-ended Visual Question Answering Guided by World Knowledge

[Generate then Select: Open-ended Visual Question Answering Guided by
World Knowledge](https://arxiv.org/pdf/2305.18842.pdf) accepted to ACL 2023 Findings.

## Prerequisites

* For the possible answer generation, we build based on the PiCA code: [An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA](https://arxiv.org/pdf/2109.05014.pdf)
* Obtain the public [OpenAI GPT API key](https://openai.com/api/) and install the [API Python bindings](https://beta.openai.com/docs/api-reference/introduction).
* For the VQA training part, we build upon the model KAT's code: [A Knowledge Augmented Transformer for Vision-and-Language](https://github.com/guilk/KAT)

## How to run the code

To start with,
```bash
git clone --recurse-submodules git@github.com:awslabs/vqa-generate-then-select.git
cd vqa-generate-then-select
cp -r src/* KAT
cp -r PICa/* KAT
cd KAT
pip install -r requirements.txt
pip install -r requirements-new.txt
pip install -e .
```

1. Candidate generation

```
python gen_answers.py
```

2. Train VQA selector

```
python build_vqa_input.py

bash train_vqa.sh
```


## References

[*Generate then Select: Open-ended Visual Question Answering Guided by World Knowledge*](https://aclanthology.org/2023.findings-acl.147.pdf)


```bibtex
@inproceedings{fu-etal-2023-generate,
title = "Generate then Select: Open-ended Visual Question Answering Guided by World Knowledge",
author = "Fu, Xingyu  and
    Zhang, Sheng  and
    Kwon, Gukyeong  and
    Perera, Pramuditha  and
    Zhu, Henghui  and
    Zhang, Yuhao  and
    Li, Alexander Hanbo  and
    Wang, William Yang  and
    Wang, Zhiguo  and
    Castelli, Vittorio  and
    Ng, Patrick  and
    Roth, Dan  and
    Xiang, Bing",
booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
month = jul,
year = "2023",
address = "Toronto, Canada",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2023.findings-acl.147" }
```