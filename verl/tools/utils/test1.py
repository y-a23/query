from search_r1_like_utils import perform_single_search_batch

perform_single_search_batch(
    "http://127.0.0.1:8000/retrieve",
    ["123","456"],
    topk=3
)
