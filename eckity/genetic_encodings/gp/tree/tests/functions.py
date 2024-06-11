def untyped_add(a, b):
    return a + b


def untyped_sub(a, b):
    return a - b


def untyped_mul(a, b):
    return a * b


def untyped_div(a, b):
    return b if b == 0 else a // b


def typed_add(a: int, b: int) -> int:
    return a + b


def typed_sub(a: int, b: int) -> int:
    return a - b


def typed_mul(a: int, b: int) -> int:
    return a * b


def typed_div(a: int, b: int) -> int:
    return b if b == 0 else a // b
