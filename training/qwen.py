import random
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from datasets import load_dataset
import torch
from qwen_vl_utils import process_vision_info
import gc
import time
from transformers import BitsAndBytesConfig
from trl import SFTConfig
import trackio
from peft import LoraConfig
from trl import SFTTrainer


model_name = "Qwen/Qwen3-VL-2B-Instruct"


def run_qwen(message=None):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # prepare the model input
    if message is None:
        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "user", "content": prompt}
        ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("content:", content)


def format_data(system_message, sample):
    return {
      "images": [sample["image"]],
      "messages": [

          {
              "role": "system",
              "content": [
                  {
                      "type": "text",
                      "text": system_message
                  }
              ],
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": sample["image"],
                  },
                  {
                      "type": "text",
                      "text": sample['query'],
                  }
              ],
          },
          {
              "role": "assistant",
              "content": [
                  {
                      "type": "text",
                      "text": sample["label"][0]
                  }
              ],
          },
      ]
      }


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample['messages'])

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text


def eval_qwen(model_name, test_dataset):

    # model and processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen3VLProcessor.from_pretrained(model_name)

    # test generation
    rand_int = random.randint(0, 10)
    sample = test_dataset[rand_int]
    # save the image
    image = sample['messages'][1]['content'][0]['image']
    image.save("test_image.png")
    output_text = generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda")
    print("output_text:", output_text)  


def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def sft_qwen_train(model_name, train_dataset, eval_dataset, test_dataset=None):

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = Qwen3VLProcessor.from_pretrained(model_name)

    if test_dataset is not None:
        rand_int = random.randint(0, 10)
        print("Running test generation before SFT...")
        test_sample = test_dataset[rand_int]
        test_image = test_sample['images'][0]
        output = generate_text_from_sample(model, processor, test_sample)
        print("  test output:", output)

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model_id = model_name.split("/")[-1]

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=model_id,  # Directory to save the model
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        max_length=None,
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=1e-5,  # Learning rate for training
        # Logging and evaluation
        logging_steps=1,  # Steps interval for logging
        eval_steps=1,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=1,  # Maximum norm for gradient clipping
        warmup_steps=5,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=True,  # Whether to push model to Hugging Face Hub
        report_to="trackio",  # Reporting tool for tracking metrics
        # gemini proposed
        gradient_checkpointing=True,  # Enable gradient checkpointing
        use_cache=False,  # Disable caching for training
    )

    trackio.init(
        project=model_id,
        name=model_id,
        config=training_args,
        space_id=training_args.output_dir + "-trackio"
    )

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    if test_dataset is not None:
        print("Running test generation after SFT...")
        test_sample = test_dataset[rand_int]
        test_image = test_sample['images'][0]
        test_image.save(f"sft_test_image_{rand_int}.png")
        output = generate_text_from_sample(model, processor, test_sample)
        print("  test output:", output)


def sft_qwen_test(model_name, test_dataset):

    clear_memory()

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    model_id = model_name.split("/")[-1]
    processor = Qwen3VLProcessor.from_pretrained(model_name)
    model.load_adapter(model_id)

    test_sample = test_dataset[0]
    test_image = test_sample['images'][0]
    test_image.save("sft_test_image.png")
    output = generate_text_from_sample(model, processor, test_sample)
    print("test sft output:", output)


if __name__ == "__main__":


    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
        Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""    
    
    # data
    dataset_id = "HuggingFaceM4/ChartQA"
    train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10]', 'val[:10]', 'test[:10]'])
    train_dataset = [format_data(system_message, sample) for sample in train_dataset]
    eval_dataset = [format_data(system_message, sample) for sample in eval_dataset]
    test_dataset = [format_data(system_message, sample) for sample in test_dataset]

    sft_qwen_train(model_name, train_dataset, eval_dataset)
    # sft_qwen_test(model_name, test_dataset)