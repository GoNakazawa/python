# __next__をで次々と値を返し，__iter__で自分自身を返すようにするにすれば
# 自作のiteratorを作ることができる
class MyIterator:
    def __init__(self, start=0):
        self.current = start

    def __next__(self):
        num = self.current
        self.current += 1
        return num

    def __iter__(self):
        return self


class EvenIterator:
    def __init__(self, start):
        self.current = start

    def __next__(self):
        if self.current < 2:
            raise StopIteration
        elif self.current % 2 == 0:
            num = self.current
            self.current -= 2
            return num
        else:
            self.current -= 1
            return self.__next__()

    def __iter__(self):
        return self



if __name__ == "__main__":
    # myiterator = MyIterator(10)
    # print(next(myiterator))
    # print(next(myiterator))
    # print(next(myiterator))

    even = EvenIterator(10)
    print(next(even))
    print(next(even))
    print(next(even))
    print(next(even))
    print(next(even))
    print(next(even))
    print(next(even))