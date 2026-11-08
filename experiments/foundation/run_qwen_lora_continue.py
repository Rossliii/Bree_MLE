from common.paths import PROJECT_ROOT
from experiments.foundation.run_qwen_lora import run


def main() -> None:
    adapter_path = PROJECT_ROOT / "reports_qwen_lora" / "adapter"
    run(
        epochs=3,
        resume_adapter_path=adapter_path,
        output_dir_name="reports_qwen_lora_continued",
    )


if __name__ == "__main__":
    main()
