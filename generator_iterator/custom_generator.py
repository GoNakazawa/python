# generator function
def myrange(stop):
    start = 0
    while start < stop:
        yield start
        start += 1


# generator iteratorを返す
mr = myrange(10)
print(mr)
print(type(mr))

# generatorはnext()に入れることで，yieldの値を返す．
print(next(mr))
# generator関数は状態を保持する(つまり，前回のnext()時，ループの途中であった場合は次のnext()時に次のループに入る)
print(next(mr))


# challenge
def even(num):
    while num!=0:
        if num % 2 == 0:
            yield num
        num -= 1


for i in even(10):
    print(i)
