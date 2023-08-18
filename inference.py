from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)
from config import get_generation_config, get_bnb_config
import torch
def load_model():
    PEFT_MODEL = "pareek-yash/falcon-7b-qlora-airport-chatbot"

    config = PeftConfig.from_pretrained(PEFT_MODEL)
    model =AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, PEFT_MODEL)
    
def model_inference(model, tokenizer, input_text):
    """
        Run inference on the provided input text using the given model and tokenizer.
        Params:
            model
            tokenizer: 
            input_text: Prompt Input assistant and human
    """
    
    DEVICE = "cuda:0"

    encoding = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=get_generation_config(),
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


prompt = f"""
<human>: give me your airport's email address
<assistant>:
""".strip()

encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
  outputs = model.generate(
      input_ids=encoding.input_ids,
      attention_mask=encoding.attention_mask,
      generation_config=generation_config,
  )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))