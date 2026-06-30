"""
LLM API 호출 및 병렬 실행을 담당한다.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


def call_llm(client, sys_prompt: str, user_prompt: str, model: str = "solar-pro") -> str:
    """단일 LLM 호출."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def run_parallel(
    fn,
    tasks_args: list,
    max_workers: int = 12,
    desc: str = "",
    on_error=None,
) -> list:
    """fn을 tasks_args의 각 인자로 병렬 실행.

    tasks_args: [(arg1, arg2, ...), ...] 형태의 튜플 리스트
    on_error: task 예외 발생 시 호출되는 콜백. None이면 예외를 다시 발생시킨다.
    반환: 완료 순서대로 수집된 결과 리스트
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, *args): args for args in tasks_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            task_args = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                if on_error is None:
                    raise
                results.append(on_error(e, task_args))
    return results
