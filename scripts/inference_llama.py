import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

experiment_dir = "models/gervasio_puigcerver"

llm = LLM(
    #model="maritaca-ai/sabia-7b",
    model="PORTULAN/gervasio-7b-portuguese-ptbr-decoder",
    enable_lora=True, 
    max_model_len=256,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.001,
    max_tokens=256
)

df = pd.read_csv("prompts/puigcerver_test.csv")

prompts = []

for index, row in df.iterrows():
    prompts.append(
    f"""
### Instrução: Você receberá o texto de uma redação extraída por um sistema de OCR. Corrija os erros presentes no texto.

### Entrada: {row['input']}

### Resposta:
    """
)

predictions = pd.DataFrame([], columns=["prediction"])

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("spelling", 1, experiment_dir + "/model/model_weights")
)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    predictions = pd.concat([predictions, pd.DataFrame([generated_text], columns=["prediction"])])
    
predictions.to_csv("predictions/gervasio_puigcerver.csv", index=False)
