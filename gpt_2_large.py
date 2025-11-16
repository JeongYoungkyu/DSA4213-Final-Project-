def main():
  base_model_name = "gpt2-large"

  tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
  
  special_tokens = {
      "pad_token": "<|pad|>",
      "additional_special_tokens": [
          "### System:",
          "### User:",
          "### Assistant:",
          "<|end|>",
      ],
  }
  tokenizer.add_special_tokens(special_tokens)
  
  model = GPT2LMHeadModel.from_pretrained(base_model_name)
  model.resize_token_embeddings(len(tokenizer))
  model.config.pad_token_id = tokenizer.pad_token_id

  lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_fc", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
  )
  
  model = get_peft_model(model, lora_config)
  data_collator = DefaultDataCollator()
  
  training_args = TrainingArguments(
      output_dir="/tmp/gpt2-large-esconv-lora",
      per_device_train_batch_size=2,
      per_device_eval_batch_size=2,
      num_train_epochs=3,
      learning_rate=5e-5,
      logging_steps=50,
      eval_strategy="steps",
      save_steps=500,
      save_total_limit=2,
      fp16=True,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_ds_train,  
      eval_dataset=tokenized_ds_val,    
      data_collator=data_collator,
  )
  
  trainer.train()
  trainer.save_model("/tmp/gpt2-large-esconv-lora")
  tokenizer.save_pretrained("/tmp/gpt2-large-esconv-lora")

  ADAPTER_DIR = "/tmp/gpt2-large-esconv-lora"

  tokenizer = GPT2TokenizerFast.from_pretrained(ADAPTER_DIR)
  base = GPT2LMHeadModel.from_pretrained("gpt2-large")
  base.resize_token_embeddings(len(tokenizer))
  model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

  def chat(user_text, max_new_tokens=25):
    prompt = (
        "### System: You are a safe, supportive, non-clinical emotional support assistant. "
        "You NEVER diagnose, name disorders, mention medications, insult anyone, or speculate about medical causes. "
        "You reply in ONE short, gentle, supportive sentence.\n"
        f"### User: {user_text}\n"
        "### Assistant:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        out = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
            pad_token_id=tokenizer.pad_token_id, 
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=False)

    reply = decoded.split("### Assistant:")[-1]
    reply = reply.split("<|end|>")[0].strip()

    for sep in [".", "!", "?"]:
        if sep in reply:
            reply = reply.split(sep, 1)[0] + sep
            break

    reply = reply.strip()

    banned = [
        "attack", "disorder", "schizophren", "bipolar", "diagnos",
        "self-harm", "suicide", "kill", "stupid", "crazy", "mentally ill",
        "panic attack"
    ]
    if any(b in reply.lower() for b in banned) or len(reply.split()) < 4:
        reply = (
            "It sounds like things have been really heavy for you, "
            "and it might help to talk with someone you trust or a professional who can support you."
        )

    return reply

  print(chat("I feel so anxious all the time."))
  print(chat("Everything feels too much."))
  print(chat("I feel empty inside."))
  print(chat("Nothing feels enjoyable anymore."))
  print(chat("I feel like no one cares about me."))
  print(chat("I’m always alone even when I’m with people."))
  print(chat("I’m so tired but I can’t stop working."))
  print(chat("I feel like I’m failing at everything."))
  print(chat("I feel like a burden to everyone."))
  print(chat("I don’t think I’m good enough."))

if __name__ == "__main__":
    main()
