from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

def test_model(prompt, max_length=1000):
    """
    Test the LoRA-finetuned Mistral model
    
    Args:
        prompt (str): Input text to generate from
        max_length (int): Maximum length of generated text
    """
    try:
        # Load base model and tokenizer
        base_model = "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,  # Use float16 to save memory
            device_map="auto"  # Automatically handle device placement
        )

        # Load and apply LoRA weights
        model = PeftModel.from_pretrained(
            model,
            "akoirala/clinical-history-eval",
            torch_dtype=torch.float16
        )

        # Prepare the input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "status": "success",
            "input_prompt": prompt,
            "generated_text": generated_text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Example usage
if __name__ == "__main__":
    # Example prompt - modify based on your use case
    test_prompt = """
### Instruction

You are an AI assistant for the department of radiology at a hospital. Your task is to extract structured information from notes provided by a clinician to the radiology department. You can use the following rubric for the kind of structured information to extract as a guide:

* Past Medical History: Refers to aspects of the patient's medical history that may be relevant to the current clinical scenario. This information is often, but not always, followed by phrases like 'hx' or 'pmh'.
* What: Refers to the most relevant signs and symptoms prompting the order for imaging.
* When: Refers to the time course of the inciting event, which may include the duration of symptoms, date of onset of illness, or time of incident or injury.
* Where: Refers to the precise localization of symptoms or anatomical site of pain or other abnormality, if relevant. For symptoms that require a specific location for diagnosis, such as pain, injury, or deformity, extract the precise anatomical location of symptoms. For symptoms are not localized to a specific anatomical region, such as cough, dizziness, or fatigue, output "not applicable". Otherwise, output "Not included".
* Clinical concern: refers to diagnostic entities that the referring clinician feels are most likely or most important to exclude. This information is often, but not always, followed by phrases like 'concern for' or 'rule out'.

### Task
Based on the information above, please structure the clinician's note provided below. Make sure all medical abbreviations in the output are expanded to their full forms. Output "Not included" if information is NOT found for certain category. 

Note 1: See comments Order History: Relevant PMH urinary retention and frequent falls. Presents with s/p fall for a duration of today. Specific location of issue (if applicable): hit head. Concern for pelvic fx.
    """
    
    print("Testing model...")
    result = test_model(test_prompt)
    
    if result["status"] == "success":
        print("\nInput prompt:")
        print(result["input_prompt"])
        print("\nGenerated text:")
        print(result["generated_text"])
    else:
        print("Error:", result["message"])

    print("\nTest completed!")