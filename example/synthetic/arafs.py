from araf import ARAF
from minheap import MinHeap


class ARAF_fs(ARAF):
    def __init__(self, n_freq, n_conf):
        super().__init__(n_freq, n_conf)
        self._heap_freq = MinHeap(n_freq)

    def count_2_itemsets(self, X, y):
        """Count 2-itemsets.
        Args:
            X: input data set, a DataFrame.
            y: labels, an n*1 DataFrame.
        Returns:
            a dict whose key is a tuple as (f1, f2, val1, val2, c),
            and the corresponding value is the frequency of the tuple.
        """
        frequent_set = {key for key, _ in self._heap_freq.items()}
        columns = list(X.columns)
        indices = {col: i for i, col in enumerate(columns)}
        X = X.values

        self._count_2 = {}
        freq_columns = list({f for f, val, c in frequent_set})
        for row, c in zip(X, y):
            for i1, f1 in enumerate(freq_columns):
                val1 = row[indices[f1]]
                if (f1, val1, c) in frequent_set:
                    for i2 in range(i1 + 1, len(freq_columns)):
                        f2 = freq_columns[i2]
                        val2 = row[indices[f2]]
                        if (f2, val2, c) in frequent_set:
                            if (f1, f2, val1, val2, c) in self._count_2:
                                self._count_2[(f1, f2, val1, val2, c)] += 1
                            else:
                                self._count_2[(f1, f2, val1, val2, c)] = 1

    def select_frequent_1_itemsets(self):
        """Select 1-itemsets that are most frequent for each class."""
        for key in self._count_1:
            self._heap_freq.push(key, self._count_1[key])

    def select_frequent_itemsets(self):
        """Select 1-itemsets and 2-itemsets that are most frequent for each class."""
        for key in self._count_2:
            self._heap_freq.push(key, self._count_2[key])

    def select_itemsets(self, X):
        """Select frequent 1-itemsets and 2-itemsets that are most confident.
           If several itemsets have the same confidence and space is not sufficient to keep all of them,
           then only those with largest support will be remained.
           If a main effect and its interaction are both confident,
           then the interaction will be remained only when it's more confident than the main effect.
        """
        columns = list(X.columns)
        indices = {columns[i]: i for i in range(len(columns))}
        X = X.values

        frequent_set = self._heap_freq.items()

        self._heap_conf = MinHeap(self.n_conf)
        for key, _ in frequent_set:
            if len(key) == 3:
                f, val, c = key
                self._heap_conf.push(key, self._count_1[key] / sum(X[:, indices[f]] == val))
            else:
                f1, f2, val1, val2, c = key
                conf = self._count_2[key] / sum((X[:, indices[f1]] == val1) & (X[:, indices[f2]] == val2))
                self._heap_conf.push(key, conf)
        self.s1 = {key[:2] for key, _ in self._heap_conf.items() if len(key) == 3}
        self.s2 = {key[:4] for key, _ in self._heap_conf.items() if len(key) == 5}


class ARAF_ub(ARAF):
    def __init__(self, n_freq, n_conf):
        super().__init__(n_freq, n_conf)

    def select_itemsets(self, X):
        """Select frequent 1-itemsets and 2-itemsets that are most confident.
           If several itemsets have the same confidence and space is not sufficient to keep all of them,
           then only those with largest support will be remained.
           If a main effect and its interaction are both confident,
           then the interaction will be remained only when it's more confident than the main effect.
        """
        columns = list(X.columns)
        indices = {columns[i]: i for i in range(len(columns))}
        X = X.values

        frequent_set = []
        for c in self._heaps_freq:
            frequent_set += self._heaps_freq[c].items()
        frequent_set.sort(key=lambda x: (-x[1], len(x[0])))

        self._heap_conf = MinHeap(self.n_conf)
        for key, _ in frequent_set:
            if len(key) == 3:
                f, val, c = key
                conf = self._count_1[key] / sum(X[:, indices[f]] == val)
                # calculate the relative confidence
                conf = conf / (1 - conf) / self._margin[c] * (1 - self._margin[c])
                self._heap_conf.push(key, conf)
            else:
                f1, f2, val1, val2, c = key
                key1, key2 = (f1, val1, c), (f2, val2, c)
                conf = self._count_2[key] / sum((X[:, indices[f1]] == val1) & (X[:, indices[f2]] == val2))
                # calculate the relative confidence
                conf = conf / (1 - conf) / self._margin[c] * (1 - self._margin[c])
                self._heap_conf.push(key, conf)
        self.s1 = {key[:2] for key, _ in self._heap_conf.items() if len(key) == 3}
        self.s2 = {key[:4] for key, _ in self._heap_conf.items() if len(key) == 5}