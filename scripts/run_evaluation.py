"""
Reproducible Benchmark Evaluation Runner for MedRAX & ChestAgentBench
---------------------------------------------------------------------
Supports:
  1. Direct MLLM Baselines (e.g. GPT-4o, Gemini, Claude)
  2. Full / Modular MedRAX Agent with Dynamic Tool Calling
  3. Resume interrupted evaluations
  4. Per-category and overall accuracy computation
"""

import os
# Force transformers to use PyTorch only (prevents TensorFlow protobuf mismatch in Colab Python 3.13)
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import sys
import json
import time
import torch
import base64
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LETTERS = ["A", "B", "C", "D", "E", "F"]
CATEGORIES_LIST = [
    "Detection",
    "Classification",
    "Localization",
    "Comparison",
    "Relationship",
    "Diagnosis",
    "Characterization",
]

def encode_image_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[Warning] Failed to read {image_path}: {e}")
        return None

def extract_answer_letter(text: str) -> Optional[str]:
    """Robust extraction of A-F choice from model response."""
    if not text:
        return None
    # 1. Match explicit pattern: Answer: [A-F] or Option: [A-F] or [A-F])
    match = re.search(r"(?:final answer|answer|choice|option)[\s:]*([A-F])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 2. Match bold **A** or (A)
    match = re.search(r"\b\**\(([A-F])\)\**\b|\b\*\*([A-F])\*\*\b", text)
    if match:
        return (match.group(1) or match.group(2)).upper()
    # 3. Match phrasing: "is [A-F]", "select [A-F]", "choose [A-F]"
    match = re.search(r"\b(?:is|select|choose|indicates|diagnosed as|diagnosis is)\s+([A-F])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 4. Match letter at the very end
    match = re.search(r"\b([A-F])[\.\)]?\s*$", text.strip())
    if match:
        return match.group(1).upper()
    # 5. Standalone letter followed by colon/dot/paren
    match = re.search(r"\b([A-F])[\.\:\)]", text)
    if match:
        return match.group(1).upper()
    # 6. First standalone letter A-F in first 50 chars
    match = re.search(r"\b([A-F])\b", text[:50])
    if match:
        return match.group(1).upper()
    return None

def get_api_credentials(requested_model: str):
    """Auto-detects credentials and configures OpenAI, Local (Ollama/vLLM), or Gemini endpoints."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", None)

    # 1. If pointing to local server (Ollama, vLLM, LM Studio), use local endpoint directly
    if base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
        return api_key or "ollama", base_url, requested_model

    # 2. If using Groq or custom provider URL, preserve it
    if base_url and ("groq" in base_url or "api." in base_url):
        return api_key, base_url, requested_model

    # 3. If no OpenAI key and no custom base_url, check for Google Gemini API key
    if not api_key or api_key.startswith("your-"):
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            api_key = gemini_key
            base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            if "gpt" in requested_model or requested_model == "chatgpt-4o-latest":
                requested_model = "gemini-3.5-flash"
            # Set into env for LangChain/LangGraph tools
            os.environ["OPENAI_API_KEY"] = gemini_key
            os.environ["OPENAI_BASE_URL"] = base_url

    return api_key, base_url, requested_model

def evaluate_baseline_gemini(example: Dict[str, Any], figures_dir: Path, model_name: str = "gemini-3.5-flash", api_key: str = None) -> Dict[str, Any]:
    """Native Google Gemini evaluation with automatic fallback across available Flash models."""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)

    prompt = (
        f"You are an expert thoracic radiologist.\n"
        f"Given the clinical case and thoracic radiographs, answer this multiple choice question.\n\n"
        f"{example['question']}\n\n"
        f"Analyze the imaging carefully. State your reasoning, then conclude with 'Answer: X' (where X is A, B, C, D, E, or F)."
    )

    parts = [prompt]
    image_paths = example.get("images", [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    loaded_imgs = 0
    for img_rel in image_paths:
        clean = img_rel.replace("figures/", "") if img_rel.startswith("figures/") else img_rel
        full_img_path = figures_dir / clean
        if full_img_path.exists():
            try:
                img = Image.open(full_img_path).convert("RGB")
                parts.append(img)
                loaded_imgs += 1
            except Exception as e:
                print(f"[Warning] Failed to load image {full_img_path}: {e}")

    # Fallback to URLs if local images not loaded
    if loaded_imgs == 0 and "image_source_urls" in example:
        import requests, io
        urls = example["image_source_urls"]
        if isinstance(urls, str):
            urls = [urls]
        for u in urls:
            if u and isinstance(u, str):
                try:
                    resp = requests.get(u, timeout=10)
                    if resp.status_code == 200:
                        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        parts.append(img)
                except Exception:
                    pass

    # Model fallback hierarchy
    candidate_models = [model_name, "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]
    candidate_models = list(dict.fromkeys([m for m in candidate_models if m]))

    raw_response = None
    last_err = None
    for cand in candidate_models:
        try:
            m = genai.GenerativeModel(cand)
            resp = m.generate_content(parts)
            raw_response = resp.text
            break
        except Exception as e:
            last_err = e
            continue

    if raw_response is None:
        raise RuntimeError(f"Gemini API generation failed across candidate models: {last_err}")

    pred_letter = extract_answer_letter(raw_response)
    return {
        "raw_response": raw_response,
        "prediction": pred_letter,
        "tools_called": [],
    }

def evaluate_baseline_mllm(example: Dict[str, Any], figures_dir: Path, model_name: str = "gpt-4o") -> Dict[str, Any]:
    api_key, base_url, effective_model = get_api_credentials(model_name)

    # Route to native Gemini if Google key is detected and not a local/custom endpoint
    is_custom_or_local = base_url and any(k in base_url for k in ["localhost", "127.0.0.1", "groq", "api."])
    if not is_custom_or_local and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")) and ("gemini" in model_name or not os.getenv("OPENAI_API_KEY")):
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return evaluate_baseline_gemini(example, figures_dir, model_name=effective_model, api_key=gemini_key)

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key or "local",
        base_url=base_url
    )

    prompt = (
        f"You are a medical reasoning assistant specialized in chest radiology.\n"
        f"Given the clinical case and thoracic radiographs, answer this multiple choice question.\n\n"
        f"{example['question']}\n\n"
        f"Analyze the imaging carefully. State your reasoning, then conclude with 'Answer: X' (where X is A, B, C, D, E, or F)."
    )

    # Check if target model supports multimodal image_url
    is_vision = not is_custom_or_local and any(k in model_name.lower() for k in ["vision", "gpt-4", "gemini", "claude", "4o", "vl"])
    content = [{"type": "text", "text": prompt}]

    if is_vision:
        # Load local images
        image_paths = example.get("images", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        loaded_imgs = 0
        for img_rel in image_paths:
            clean = img_rel.replace("figures/", "") if img_rel.startswith("figures/") else img_rel
            full_img_path = figures_dir / clean
            if full_img_path.exists():
                b64 = encode_image_base64(str(full_img_path))
                if b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
                    loaded_imgs += 1

        # Fallback to URLs if local images not found
        if loaded_imgs == 0 and "image_source_urls" in example:
            urls = example["image_source_urls"]
            if isinstance(urls, str):
                urls = [urls]
            for u in urls:
                if u and isinstance(u, str):
                    content.append({"type": "image_url", "image_url": {"url": u}})

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content if is_vision else prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    raw_response = response.choices[0].message.content
    pred_letter = extract_answer_letter(raw_response)

    return {
        "raw_response": raw_response,
        "prediction": pred_letter,
        "tools_called": [],
    }

def evaluate_medrax_agent(example: Dict[str, Any], figures_dir: Path, agent: Any, thread_id: str = "1", model_name: str = "") -> Dict[str, Any]:
    from langchain_core.messages import HumanMessage
    
    image_paths = example.get("images", [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    figure_prompt = "Relevant image files:\n"
    content_list = []
    valid_map = {}
    
    for img_rel in image_paths:
        clean = img_rel.replace("figures/", "") if img_rel.startswith("figures/") else img_rel
        full_img_path = figures_dir / clean
        if full_img_path.exists():
            figure_prompt += f"- Local image path: {full_img_path}\n"
            valid_map[Path(clean).name] = str(full_img_path)
            valid_map[str(clean)] = str(full_img_path)
            valid_map[str(img_rel)] = str(full_img_path)
            b64 = encode_image_base64(str(full_img_path))
            if b64:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

    agent.current_case_images = valid_map

    instruction = (
        f"{figure_prompt}\n"
        f"Solve the following multiple-choice clinical question using chain of thought reasoning.\n"
        f"You have access to specialized chest radiology tools (classification, segmentation, VQA, report generation).\n"
        f"Use tools whenever appropriate to verify findings.\n\n"
        f"{example['question']}\n\n"
        f"End your final clinical answer explicitly with: 'Answer: X' (where X is A, B, C, D, E, or F)."
    )

    base_url_str = (os.getenv("OPENAI_BASE_URL") or "").lower()
    is_text_only = any(k in base_url_str for k in ["groq", "localhost", "127.0.0.1"]) or any(k in model_name.lower() for k in ["oss", "qwen2.5", "llama-3"])
    is_vision = not is_text_only and any(k in model_name.lower() for k in ["vision", "gpt-4", "gemini", "claude", "4o", "vl"])
    if is_vision and content_list:
        messages = [HumanMessage(content=[{"type": "text", "text": instruction}] + content_list)]
    else:
        messages = [HumanMessage(content=instruction)]

    thread = {"configurable": {"thread_id": str(thread_id)}}

    final_content = ""
    tools_used = []

    for event in agent.workflow.stream({"messages": messages}, thread):
        for node_name, val in event.items():
            if isinstance(val, dict) and "messages" in val:
                for msg in val["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools_used.append(tc.get("name", "tool"))
                    # ONLY record response from AIMessage text (never from ToolMessage)
                    if msg.__class__.__name__ == "AIMessage" and msg.content and isinstance(msg.content, str) and msg.content.strip():
                        final_content = msg.content

    # If final_content is still empty or tool message, retrieve last AIMessage from full thread history
    if not final_content:
        state = agent.workflow.get_state(thread)
        if state and hasattr(state, "values") and "messages" in state.values:
            for msg in reversed(state.values["messages"]):
                if msg.__class__.__name__ == "AIMessage" and msg.content and isinstance(msg.content, str) and msg.content.strip():
                    final_content = msg.content
                    break

    pred_letter = extract_answer_letter(final_content)
    return {
        "raw_response": final_content,
        "prediction": pred_letter,
        "tools_called": list(set(tools_used)),
    }

def run_benchmark(
    data_dir: str,
    output_dir: str,
    mode: str = "baseline",
    model_name: str = "chatgpt-4o-latest",
    max_cases: Optional[int] = None,
    tool_list: Optional[List[str]] = None,
    device: str = "cuda",
    resume: bool = True,
    reset: bool = False,
):
    data_path = Path(data_dir).resolve()
    metadata_file = data_path / "metadata.jsonl"
    figures_dir = data_path / "figures"
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = out_path / "predictions.jsonl"
    summary_file = out_path / "summary.json"

    # Reset or Resume handling
    done_ids = set()
    existing_preds = []
    if reset:
        print(f"[Reset] --reset active: Clearing previous predictions in {out_path} to evaluate 100% clean under {model_name}.")
        if pred_file.exists():
            pred_file.unlink()
        if summary_file.exists():
            summary_file.unlink()
    elif resume and pred_file.exists():
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    done_ids.add(str(item.get("question_id", "")))
                    existing_preds.append(item)
        print(f"[Resume] Found {len(done_ids)} questions already evaluated. Skipping them.")

    # Load questions
    rows = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Case limiting
    if max_cases is not None:
        seen_cases = set()
        filtered = []
        for r in rows:
            cid = str(r.get("case_id", ""))
            if cid not in seen_cases and len(seen_cases) >= max_cases:
                continue
            seen_cases.add(cid)
            filtered.append(r)
        rows = filtered
        print(f"[Filter] Selected {len(rows)} questions across {len(seen_cases)} cases (limit: {max_cases}).")

    api_key, base_url, effective_model = get_api_credentials(model_name)
    model_name = effective_model

    agent = None
    if mode == "medrax":
        from main import initialize_agent
        print(f"[Init] Initializing MedRAX Agent with tools: {tool_list} on {device}...")
        prompt_file = PROJECT_ROOT / "medrax" / "docs" / "system_prompts.txt"
        model_dir = os.getenv("MEDRAX_MODEL_DIR", str(PROJECT_ROOT / "checkpoints"))
        agent, _ = initialize_agent(
            prompt_file=str(prompt_file),
            tools_to_use=tool_list,
            model_dir=model_dir,
            temp_dir=str(PROJECT_ROOT / "temp"),
            device=device,
            model=model_name,
        )

    print(f"\n=======================================================")
    print(f" Starting ChestAgentBench Evaluation")
    print(f" Mode: {mode} | Model: {model_name} | Total Questions: {len(rows)}")
    print(f" Results will be saved to: {pred_file}")
    print(f"=======================================================\n")

    correct = 0
    total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    # Accumulate previous results if resuming
    for p in existing_preds:
        ans = p.get("gold_answer", "").strip().upper()
        pred = (p.get("prediction") or "").strip().upper()
        is_corr = (pred == ans)
        total += 1
        if is_corr:
            correct += 1
        for cat in p.get("categories", []):
            cat_total[cat] += 1
            if is_corr:
                cat_correct[cat] += 1

    BATCH_SIZE = 10  # Flush to Google Drive every N questions
    pending_records = []

    def flush_batch(records):
        """Append a batch of records to the prediction file and force-sync to Drive."""
        if not records:
            return
        with open(pred_file, "a", encoding="utf-8") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

    for idx, row in enumerate(rows, 1):
        qid = str(row.get("question_id", ""))
        if qid in done_ids:
            continue

        gold_ans = str(row.get("answer", "")).strip().upper()
        raw_cats = row.get("categories", "")
        cats = [c.strip() for c in (raw_cats if isinstance(raw_cats, list) else str(raw_cats).split(",")) if c.strip()]

        try:
            res = None
            for attempt in range(3):
                try:
                    if mode == "medrax":
                        res = evaluate_medrax_agent(row, figures_dir, agent, thread_id=f"t_{idx}_{attempt}", model_name=model_name)
                    else:
                        res = evaluate_baseline_mllm(row, figures_dir, model_name=model_name)
                    break
                except Exception as e_inner:
                    if attempt < 2 and any(k in str(e_inner).lower() for k in ["connection", "connect", "timeout"]):
                        time.sleep(3 * (attempt + 1))
                        continue
                    raise e_inner

            pred_letter = res.get("prediction")
            is_correct = (pred_letter == gold_ans)

            total += 1
            if is_correct:
                correct += 1

            for cat in cats:
                cat_total[cat] += 1
                if is_correct:
                    cat_correct[cat] += 1

            record = {
                "question_id": qid,
                "case_id": row.get("case_id"),
                "gold_answer": gold_ans,
                "prediction": pred_letter,
                "is_correct": is_correct,
                "categories": cats,
                "tools_called": res.get("tools_called", []),
                "raw_response": res.get("raw_response", ""),
                "timestamp": datetime.now().isoformat(),
            }
            pending_records.append(record)

            curr_acc = (correct / total) * 100 if total > 0 else 0
            status_symbol = "✓" if is_correct else "✗"
            print(f"[{idx}/{len(rows)}] QID: {qid} | Pred: {pred_letter} Gold: {gold_ans} {status_symbol} | Running Acc: {curr_acc:.2f}% ({correct}/{total})")

            # Batch commit: force-sync every BATCH_SIZE questions to Google Drive
            if len(pending_records) >= BATCH_SIZE:
                flush_batch(pending_records)
                pending_records = []
                print(f"  [Drive Sync] Flushed batch to Google Drive at question {idx}.")

        except Exception as e:
            print(f"[Error processing QID {qid}]: {e}")
            # Flush any pending records before sleeping to avoid data loss
            if pending_records:
                flush_batch(pending_records)
                pending_records = []
            time.sleep(2)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final flush for any remaining records
    if pending_records:
        flush_batch(pending_records)
        print(f"  [Drive Sync] Final flush: {len(pending_records)} remaining records saved to Google Drive.")

    # Compute Final Summary Metrics
    overall_acc = (correct / total) * 100 if total > 0 else 0.0
    category_summary = {}
    for cat in CATEGORIES_LIST:
        c_tot = cat_total.get(cat, 0)
        c_corr = cat_correct.get(cat, 0)
        category_summary[cat] = {
            "correct": c_corr,
            "total": c_tot,
            "accuracy": round((c_corr / c_tot) * 100, 2) if c_tot > 0 else 0.0
        }

    summary = {
        "mode": mode,
        "model": model_name,
        "total_evaluated": total,
        "total_correct": correct,
        "overall_accuracy": round(overall_acc, 2),
        "category_metrics": category_summary,
        "completion_time": datetime.now().isoformat(),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print(f" Evaluation Completed!")
    print(f" Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f" Category Breakdown:")
    for cat, data in category_summary.items():
        print(f"   - {cat:<18}: {data['accuracy']:>6.2f}% ({data['correct']}/{data['total']})")
    print(f" Detailed results: {pred_file}")
    print(f" Summary metrics:  {summary_file}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedRAX on ChestAgentBench")
    parser.add_argument("--mode", choices=["medrax", "baseline"], default="baseline")
    parser.add_argument("--model", type=str, default="chatgpt-4o-latest")
    parser.add_argument("--api-key", type=str, default=None, help="API Key (Groq, OpenAI, Gemini)")
    parser.add_argument("--base-url", type=str, default=None, help="Custom Base URL (e.g. https://api.groq.com/openai/v1)")
    parser.add_argument("--data-dir", type=str, default=os.getenv("BENCHMARK_DATA_DIR", "./data/chestagentbench"))
    parser.add_argument("--output-dir", type=str, default=os.getenv("MEDRAX_RESULTS_DIR", "./results"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Clear all previous predictions to evaluate completely fresh (100%%) under this model")
    args = parser.parse_args()

    # Apply explicit CLI credentials if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    default_tools = [
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
    ]

    run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        model_name=args.model,
        max_cases=args.max_cases,
        tool_list=default_tools,
        device=args.device,
        resume=(not args.no_resume and not args.reset),
        reset=args.reset,
    )
