from swift.llm import PtEngine, RequestConfig, InferRequest

if __name__ == "__main__":
    model_id = "deepseek-ai/deepseek-vl2-tiny"
    prompt = "<image> Describe the image."
    adapter = None

    engine = PtEngine(model_id, max_batch_size=2, use_hf=True, adapters=adapter)
    request_config = RequestConfig(max_tokens=128, temperature=0,
                                   top_k=None)

    messages = [
        {"role": "system",
         "content": "You are a professional anomaly detection and classification tool that detects objects that prevent an excavator from digging. You will be first presented with some example images of the trench and the bucket. Then you will be asked to detect anomalies in the trench."},
        {"role": "assistant", "content": ""},
        {"role": "user",
         "content": "<image> This image shows the trench with the excavator's bucket. Use the bucket size to check if stones are too big to fit and should count as anomalies."},
        {"role": "assistant", "content": ""},
        {"role": "user",
         "content": "<image> Describe all images you see."},
    ]

    images = ["/home/louis/workspace/Anomaly_Det_VLM/context/bucket.png", "/home/louis/workspace/Anomaly_Det_VLM/0_0.png"]

    infer_request = InferRequest(messages=messages, images=images)

    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f"response: {response}")
