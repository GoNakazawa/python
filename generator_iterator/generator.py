import sys
import csv
# 以下はどちらも同じ結果だが，
# range()はgeneratorを返すので全ての数字をメモリにはのせるわけではない
range_nums = range(10)
for i in range_nums:
    print(i)
# listは全てのデータがメモリにのる
list_nums = list(range(10))
for i in list_nums:
    print(i)

# sys.getsizeofでメモリ使用量を確認すると，listの方が大きいことがわかる
print(sys.getsizeof(range_nums))
print(sys.getsizeof(list_nums))

# generatorは大きなデータを扱うときに効果を発揮する．
# csvファイルのReaderもgeneratorである．
with open("example.csv") as f:
    reader = csv.DictReader(f)
    print(sys.getsizeof(reader))
    for line in reader:
        print(line)
