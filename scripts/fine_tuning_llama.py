import pandas as pd
import logging
from ludwig.api import LudwigModel

#model = "maritaca-ai/sabia-7b"
model = 'PORTULAN/gervasio-7b-portuguese-ptbr-decoder'
dataset = 'puigcerver'

df = pd.read_csv(f"prompts/{dataset}_train.csv")
config = {
    "model_type": "llm",
        "base_model": model,
        "input_features": [
            {
                "name": "input",
                "type": "text"
            }
        ],
        "output_features": [
            {
                "name": "output",
                "type": "text"
            }
        ],
        "prompt": {
            "template": "### Instrução: Você receberá o texto de uma redação extraída por um sistema de OCR. Corrija os erros presentes no texto. \n### Entrada: {input}\n### Resposta:"
        },
        "generation": {
            "temperature": 0.001,
            "max_new_tokens": 256
        },
        "adapter": {
            "type": "lora"
        },
        "quantization": {
            "bits": 4
        },
        "preprocessing": {
            "split": {
                "type": "random",
                "probabilities": [
                    0.9,
                    0.1,
                    0
                ]
            }
        },
        "trainer": {
            "type": "finetune",
            "epochs": 2,
            "batch_size": 2,
            "eval_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.0001,
            "learning_rate_scheduler": {
                "warmup_fraction": 0.03
            }
        }
}
model = LudwigModel(config=config, logging_level=logging.INFO)
model.train(dataset=df)
