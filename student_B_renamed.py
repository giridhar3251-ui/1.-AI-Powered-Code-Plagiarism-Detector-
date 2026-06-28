def sortBubble(data):
    length = len(data)
    for x in range(length):
        for y in range(0, length - x - 1):
            if data[y] > data[y + 1]:
                data[y], data[y + 1] = data[y + 1], data[y]
    return data


def fib(num):
    if num <= 1:
        return num
    first, second = 0, 1
    for _ in range(2, num + 1):
        first, second = second, first + second
    return second


def checkPalindrome(text):
    clean_text = text.lower().replace(" ", "")
    return clean_text == clean_text[::-1]
