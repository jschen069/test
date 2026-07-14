from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8005,
        url="",
        max_out_len=65536,
        batch_size=8,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            seed=0,
            repetition_penalty=1.0,
            chat_template_kwargs=dict(
                enable_thinking=False,
            ),
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
