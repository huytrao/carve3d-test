# Checkpoint download links

Không có checkpoint nhị phân nào được commit vào repository này. Các file
`.safetensors` đã được ignore bởi Git và Kaggle tải chúng vào thư mục làm việc
của notebook, không phải vào GitHub.

| Model | Dùng cho | Link tải / model hub | Cách pipeline tải |
| --- | --- | --- | --- |
| LGM `model_fp16_fixrot.safetensors` | Tái tạo Gaussian 3D, render và MRC | [Tải trực tiếp (830 MB)](https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors) · [Model page](https://huggingface.co/ashawkey/LGM) | `scripts/setup_lgm_kaggle.sh` tải vào `<LGM_ROOT>/pretrained/` và xác minh SHA-256 `744d6324656342c64f871308e73db97f0eb51858d94329b30090e986a6d050ab`. |
| MVDream `ashawkey/mvdream-sd2.1-diffusers` | Prompt → 4 ảnh nhất quán | [Model page / download](https://huggingface.co/ashawkey/mvdream-sd2.1-diffusers) | `from_pretrained()` tự tải qua Hugging Face cache trong lúc chạy `--prompt`; không có file checkpoint nào trong Git. |

Trong Kaggle, chỉ cần bật **Internet**. Lần chạy đầu của
`kaggle_full_pipeline_prompt.py` tự chạy setup LGM; lần sau tái sử dụng các
file đã tải trong `/kaggle/working/LGM` và Hugging Face cache. Nếu muốn tải
thủ công, dùng link trực tiếp ở bảng trên rồi truyền `--checkpoint /path/to/model_fp16_fixrot.safetensors`.
