import sys
# リスト内包表記のように書く(括弧は[]の代わりに()を使用)
square_gen = (num * num for num in range(10))
print(next(square_gen))
print(next(square_gen))
print(next(square_gen))
# リスト内包表記(list comprehension)との違い
square_list = [num * num for num in range(10)]
# リスト内包表記はリストなので，全てのデータがメモリにのっている
print(square_list)
print(sys.getsizeof(square_list))
# generator expressionはgeneratorなので全てのデータがメモリにあるわけではない
# next()のたびに計算される
print(square_gen)
print(sys.getsizeof(square_gen))

# conditionを使ってfilterすることも可能(リスト内包表記同様)
even_squares = (x * x for x in range(10) if x % 2 == 0)
for num in even_squares:
    print(num)
