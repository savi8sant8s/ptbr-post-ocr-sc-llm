# A proposal for post-OCR spelling correction using Language Models

**Fine-tuned models**: [HuggingFace](https://huggingface.co/savi8sant8s/ptbr-post-ocr-sc-llm)<br>
**Library for create synthetic OCR errors**: [NoisOCR](https://github.com/savi8sant8s/noisocr)

#### Abstract:
This work explores the use of Language Models (LMs) to correct residual errors in texts extracted by OCR and HTR (Handwritten Text Recognition) systems. We propose a general approach but utilize the images from Brazilian handwritten essays of the BRESSAY dataset as a use case. Two standard LMs (Bart and ByT5) and two LLMs (LLama 1 and LLama 2) were evaluated in this context. The results indicate that the smaller LMs outperformed the LLMs in terms of error rate reduction (CER and WER). Traditional correction methods, such as Symspell and Norvig, were influential in some cases but fell short of the results obtained by the LMs. ByT5 with byte-level tokenization improved CER and WER, proving performance for texts with high noise. As a result, smaller LMs, after fine-tuning, are more efficient and cheaper for post-OCR corrections. We identify and propose promising future studies involving correction at broader levels of context, such as paragraphs.

### Methodology:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/653d44f677c2f094529d86ec/COjrC4jb_CQP7IzIORGuv.png)

### Results:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/653d44f677c2f094529d86ec/cZQ1FfXvpb6SSeiAVyf1W.png)

#### Citation:
```bibtex
@inproceedings{
  araujo2024a,
  title={A proposal for post-{OCR} spelling correction using Language Models},
  author={S{\'a}vio Santos de Ara{\'u}jo and Byron Leite Dantas Bezerra and Arthur Flor de Sousa Neto and Cleber Zanchettin},
  booktitle={Latinx in AI @ NeurIPS 2024},
  year={2024},
  url={https://openreview.net/forum?id=p5P9R9AKr5}
}
```
