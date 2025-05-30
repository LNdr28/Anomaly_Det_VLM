import argparse

from swift.llm import PtEngine, RequestConfig, InferRequest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)

    args = parser.parse_args()
    image = args.img

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter = ["/ws/Anomaly_Det_VLM/configs/training_splits/Qwen/Qwen2.5-VL-3B-Instruct_train_4_16_nostone/checkpoint-150"]

    engine = PtEngine(model_id, max_batch_size=2, use_hf=True, adapters=adapter)
    request_config = RequestConfig(max_tokens=128, temperature=0,
                                   top_k=None)

    messages = [
        {"role": "system",
         "content": "You are a professional anomaly detection and classification tool that detects objects that prevent an excavator from digging. You will be first presented with some example images of the trench and the bucket. Then you will be asked to detect anomalies in the trench."},
        {"role": "assistant", "content": ""},
        {"role": "user",
         "content": "<image> This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list ’[]’. It is more important to not miss an anomaly than detecting a false positive!"},
    ]

    images = [image]

    infer_request = InferRequest(messages=messages, images=images)

    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f"response: {response}")
