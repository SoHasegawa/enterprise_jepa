import importlib
import sys
import types


def install_llm_import_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))

    ollama = types.ModuleType("ollama")
    ollama.chat = lambda *args, **kwargs: None
    ollama.Client = type("Client", (), {"__init__": lambda self, *args, **kwargs: None})
    monkeypatch.setitem(sys.modules, "ollama", ollama)

    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = type("Anthropic", (), {})
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)

    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.open = lambda *args, **kwargs: None
    pil.Image = image
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image)

    openai = types.ModuleType("openai")
    openai.OpenAI = type("OpenAI", (), {"__init__": lambda self, *args, **kwargs: None})
    monkeypatch.setitem(sys.modules, "openai", openai)

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def load_llm_module(monkeypatch):
    install_llm_import_stubs(monkeypatch)
    sys.modules.pop("src.llm", None)
    return importlib.import_module("src.llm")


def make_llm_instance(llm_module, method):
    instance = llm_module.LLM.__new__(llm_module.LLM)
    instance.method = method
    instance.multimodal = False
    instance.ollama = False
    instance.gpt5_endpoint = "gpt5-endpoint"
    instance.gpt5_mini_endpoint = "gpt5-mini-endpoint"
    return instance


def test_gpt5_call_dispatch_does_not_pass_system_prompt_as_image_paths(monkeypatch):
    llm_module = load_llm_module(monkeypatch)
    calls = []

    def fake_generate_gpt5(self, prompt, image_paths=None, temperature=0.0, endpoint=None):
        calls.append(
            {
                "prompt": prompt,
                "image_paths": image_paths,
                "temperature": temperature,
                "endpoint": endpoint,
            }
        )
        return "generated"

    monkeypatch.setattr(llm_module.LLM, "_generate_gpt5", fake_generate_gpt5)

    gpt5 = make_llm_instance(llm_module, "gpt5")
    gpt5_mini = make_llm_instance(llm_module, "gpt5-mini")

    assert gpt5("prompt", "system", image_paths=["image.png"], temperature=0.4) == "generated"
    assert gpt5_mini("prompt", "system", image_paths=None, temperature=0.2) == "generated"
    assert calls == [
        {
            "prompt": "prompt",
            "image_paths": ["image.png"],
            "temperature": 0.4,
            "endpoint": "gpt5-endpoint",
        },
        {
            "prompt": "prompt",
            "image_paths": None,
            "temperature": 0.2,
            "endpoint": "gpt5-mini-endpoint",
        },
    ]


def test_gpt5_generate_format_dispatch_uses_keyword_image_paths(monkeypatch):
    llm_module = load_llm_module(monkeypatch)
    calls = []

    def fake_generate_gpt5(self, prompt, image_paths=None, temperature=0.0, endpoint=None):
        calls.append((prompt, image_paths, temperature, endpoint))
        return "OK"

    monkeypatch.setattr(llm_module.LLM, "_generate_gpt5", fake_generate_gpt5)

    gpt5 = make_llm_instance(llm_module, "gpt5")
    gpt5_mini = make_llm_instance(llm_module, "gpt5-mini")

    assert gpt5.generate_format("prompt", "system", image_paths=["a.png"], temperature=0.1) == "OK"
    assert gpt5_mini.generate_format("prompt", "system", image_paths=["b.png"], temperature=0.3) == "OK"
    assert calls == [
        ("prompt", ["a.png"], 0.1, "gpt5-endpoint"),
        ("prompt", ["b.png"], 0.3, "gpt5-mini-endpoint"),
    ]
