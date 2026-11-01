"""HTTP-проверка запущенного PharmaQC: штатный, граничный и ошибочный сценарии.

  uvicorn app.main:app --port 8000
  python scripts/smoke_test.py --output data/eval/http_smoke_results.json
"""
import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def call(base_url, method, path, payload=None):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base_url + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="data/eval/http_smoke_results.json")
    args = parser.parse_args()

    scenarios = [
        {"id": "health", "method": "GET", "path": "/health", "expected": 200},
        {"id": "normal", "method": "POST", "path": "/api/v1/ask", "expected": 200,
         "payload": {"question": "Как маркируется сырьё, находящееся в карантине?"}},
        {"id": "no_answer", "method": "POST", "path": "/api/v1/ask", "expected": 200,
         "payload": {"question": "Какая максимальная температура хранения инъекционных растворов?"}},
        {"id": "validation", "method": "POST", "path": "/api/v1/ask", "expected": 422,
         "payload": {"question": "   "}},
    ]
    results = []
    for scenario in scenarios:
        status, body = call(args.base_url, scenario["method"], scenario["path"], scenario.get("payload"))
        passed = status == scenario["expected"]
        if scenario["id"] == "health":
            passed = passed and body.get("ready") is True
        if scenario["id"] == "normal":
            passed = passed and not body.get("refused", True) and bool(body.get("sources"))
        if scenario["id"] == "no_answer":
            passed = passed and body.get("refused") is True and body.get("sources") == []
        result = {k: v for k, v in scenario.items() if k != "payload"}
        result.update(status="PASS" if passed else "FAIL", http_status=status,
                      answer=body.get("answer"), sources=body.get("sources"), ready=body.get("ready"))
        results.append(result)
        print(f"{result['status']:4} {scenario['id']:<10} HTTP {status}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "engine": "local",
        "passed": all(r["status"] == "PASS" for r in results),
        "scenarios": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результат: {output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
