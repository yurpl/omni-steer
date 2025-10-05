def test_imports():
    import src.qwen_omni_rep_eng as pkg
    from src.qwen_omni_rep_eng.models.omni import OmniRunner
    assert OmniRunner is not None
