def calculate_factorial(num):
    result = 1
    for i in range(1, num + 1):
        result *= i
    return result


def find_max_in_list(numbers):
    maximum = numbers[0]
    for number in numbers:
        if number > maximum:
            maximum = number
    return maximum


def count_vowels(word):
    vowels = "aeiou"
    count = 0
    for letter in word.lower():
        if letter in vowels:
            count += 1
    return count
