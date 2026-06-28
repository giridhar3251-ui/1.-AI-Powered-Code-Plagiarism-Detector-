def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def fibonacci(n):
    # restructured to use recursion instead of iteration -- same logic,
    # different control-flow shape. Catching this is the job of the
    # semantic/embedding layer, not the pure AST-sequence matcher.
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def is_palindrome(s):
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
