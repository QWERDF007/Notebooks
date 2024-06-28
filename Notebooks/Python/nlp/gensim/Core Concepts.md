# [Core Concepts](https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-auto-examples-core-run-core-concepts-py)

本教程介绍了文档、语料库、向量和模型：理解和使用 gensim 所需的基本概念和术语。

```python
import pprint
```

`gensim` 的核心概念是：

1. [Document](#Document)：一些文本。
2. [Corpus](#Corpus)：文档集合。 
3. [Vector](#Vector)：文档的数学方便表示形式。 
4. [Model](#Model)：一种用于将向量从一种表示形式转换为另一种表示形式的算法。

让我们更详细地研究一下每个概念。

## Document

在 Gensim 中，一个 *document* 是一个文本序列类型（在 Python 3 中称为 str）的对象。 文档可以是任何内容，例如一条 140 个字符的推文、单个段落（即期刊文章摘要）、新闻文章或一本书。

```python
document = "Human machine interface for lab abc computer applications"
```

## Corpus

一个 *corpus* 是 Document 对象的集合。语料库在 Gensim 中扮演着两个角色：

1. 训练模型的输入。 在训练过程中，模型使用此训练语料库查找常见主题和主题，初始化其内部模型参数。 

   Gensim 专注于无监督模型，因此不需要人工干预，例如昂贵的注释或手动标记文档。

2. 用于组织的文档。 训练后，主题模型可以用于从新文档（训练语料库中未见过的文档）中提取主题。 这样的语料库可以被索引以进行[相似性查询](https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py)，通过语义相似性查询，聚类等。

这是一个示例语料库。 它包含 9 个文档，每个文档都是一个由单个句子组成的字符串。

```python
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]
```

> 重要提示
>
> 上面示例将整个语料库加载到内存中。实际操作中，语料库可能会非常庞大，因此将它们加载到内存中可能无法实现。Gensim 通过一次处理一个文档的方式智能地处理此类语料库。有关详细信息，请参阅 [Corpus Streaming – One Document at a Time](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-tutorial)。

这是一个用于说明目的的特别小的语料库示例。另一个例子可以是莎士比亚写的所有戏剧列表、所有维基百科文章列表，或者特定感兴趣的人的所有推文。

在收集语料库之后，我们通常需要进行一些预处理步骤。我们将保持简单，只删除一些常用英语单词（例如“the”）和仅在语料库中出现一次的单词。在此过程中，我们将对数据进行分词。分词将文档分解成单词（在这种情况下，使用空格作为分隔符）。

> 重要提示
>
> 执行预处理还有比单纯小写和空格分词更好的方法。有效的预处理超出了本教程的范围：如果您感兴趣，请查看 `gensim.utils.simple_preprocess()` 函数。

```python
# 创建一组常用词
stoplist = set('for a of the and to in'.split(' '))
# 将每个文档转换为小写, 按空白格分割并过滤掉停用词
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# 计算每个字的频率
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# 只保留出现超过一次的词
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)
```

输出：

```python
[['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]
```

在继续之前，我们想要将语料库中的每个词与一个唯一的整数值 ID 关联起来。 我们可以使用 `gensim.corpora.Dictionary` 类来实现这一点。 这个字典定义了我们处理所知道的词汇表中的所有单词。

```python
from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
```

输出：

```python
Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)
```

## Vector

为了推断语料库中的潜在结构，我们需要一种可以用数学方式处理的文档表示方法。一种方法是将每个文档表示为特征向量。例如，单个特征可以被认为是一个问答对：

1. 单词 *splonge* 在文档中出现了几次？ 零次。 
2. 文档由多少个段落组成？ 两个。
3.  文档使用了多少种字体？ 五种。

问题通常仅由其整数标识 (例如 1、2 和 3) 表示。然后，该文档的表示形式变成了一系列类似于 `(1, 0.0), (2, 2.0), (3, 5.0)` 的对。这称为稠密 (dense) 向量，因为它包含了上面每个问题的明确答案。

如果我们预先知道所有问题，我们可以省略它们，简单地将文档表示为 `(0, 2, 5)`。这个答案序列就是我们文档的向量（在本例中是 3 维的稠密向量）。出于实用目的，Gensim 只允许答案为 (或可以转换为) 单个浮点值的问题。

在实践中，向量通常包含许多零值。为了节省内存，Gensim 省略了所有值为 0.0 的向量元素。因此，上面的例子变成了 `(2, 2.0), (3, 5.0)`。这称为稀疏 (sparse) 向量或词袋 (bag-of-words) 向量。稀疏表示中所有缺失特征的值都可以明确地解析为零，即 `0.0`。

假设问题相同，我们可以将两个不同文档的向量进行比较。例如，假设我们给定两个向量 `(0.0, 2.0, 5.0)` 和 `(0.1, 1.9, 4.9)`。由于向量彼此非常相似，我们可以得出结论，对应于这些向量的文档也相似。当然，该结论的正确性取决于我们一开始选择问题的质量。

另一种将文档表示为向量的模型是词袋 (bag-of-words) 模型。在词袋模型下，每个文档都由一个向量表示，该向量包含词典中每个单词的出现频率计数。例如，假设我们有一个包含单词 `['coffee', 'milk', 'sugar', 'spoon']` 的词典。然后，由字符串 `"coffee milk coffee"` 组成的文档将由向量 `[2, 1, 0, 0]` 表示，其中向量的条目（按顺序）是文档中 "coffee"、"milk"、"sugar" 和 "spoon" 的出现次数。向量的长度是词典中的条目数。词袋模型的一个主要特性是它完全忽略了编码文档中令牌的顺序，这就是词袋 (bag-of-words) 这个名称的由来。

我们处理后的语料库中有 12 个唯一单词，这意味着在词袋模型下，每个文档将由一个 12 维向量表示。我们可以使用词典将标记化的文档转换为这些 12 维向量。我们可以看看这些 ID 对应的内容：

```python
pprint.pprint(dictionary.token2id)
```

输出：

```python
{'computer': 0,
 'eps': 8,
 'graph': 10,
 'human': 1,
 'interface': 2,
 'minors': 11,
 'response': 3,
 'survey': 4,
 'system': 5,
 'time': 6,
 'trees': 9,
 'user': 7}
```

例如，假设我们想对短语 "Human computer interaction" 进行向量化（请注意，此短语不在我们原始语料库中）。我们可以使用 `Dictionary` 的 `doc2bow` 方法创建文档的词袋表示，该方法返回词频的稀疏表示：

```python
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)
```

输出：

```python
[(0, 1), (1, 1)]
```

每个元组中的第一个条目对应于词典中令牌的 ID，第二个条目对应于该令牌的计数。

请注意，"interaction" 没有出现在原始语料库中，因此它没有包含在向量化中。另请注意，此向量只包含实际出现在文档中的单词的条目。由于任何给定文档只会包含词典中的少数单词，因此没有出现在向量化中的单词会隐式表示为零，以节省空间。

我们可以将整个原始语料库转换为向量列表：

```python
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)
```

输出：

```python
[[(0, 1), (1, 1), (2, 1)],
 [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
 [(2, 1), (5, 1), (7, 1), (8, 1)],
 [(1, 1), (5, 2), (8, 1)],
 [(3, 1), (6, 1), (7, 1)],
 [(9, 1)],
 [(9, 1), (10, 1)],
 [(9, 1), (10, 1), (11, 1)],
 [(4, 1), (10, 1), (11, 1)]]
```

 请注意，虽然此列表完全驻留在内存中，但在大多数应用程序中，您将需要更可扩展的解决方案。幸运的是，`gensim` 允许您使用任何一次返回单个文档向量的迭代器。有关详细信息，请参阅文档。

> 重要提示
>
> 文档和向量之间的区别在于，前者是文本，后者是文本的一种数学上的方便表示形式。有时，人们会互换使用这些术语：例如，给定某个任意文档 `D`，他们不会说“对应于文档 `D` 的向量”，而是只说“向量 `D`”或“文档 `D`”。这以产生歧义为代价实现了简洁性。
>
> 只要您记住文档存在于文档空间中，向量存在于向量空间中，那么上述歧义是可以接受的。

> 重要提示
>
> 根据获得表示的方式，两个不同的文档可能具有相同的向量表示。

## Model

现在我们已经对语料库进行了向量化，就可以开始使用 *models* 对其进行转换了。我们将模型用作一个抽象术语，指*从一种文档表示形式到另一种文档表示形式的转换*。在 `gensim` 中，文档表示为向量，因此模型可以被认为是在两个向量空间之间的转换。模型在训练过程中学习这种转换的细节，即它读取训练语料库时。

一个简单的模型示例是 [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)。tf-idf 模型将向量从词袋表示转换为一个向量空间，其中频率计数根据语料库中每个单词的相对稀有性进行加权。

这是一个简单的例子。让我们初始化 tf-idf 模型，使用我们的语料库训练它并转换字符串 "system minors":

```python
from gensim import models

# 训练模型
tfidf = models.TfidfModel(bow_corpus)

# 转换字符串 "system minors"
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])
```

输出：

```python
[(5, 0.5898341626740045), (11, 0.8075244024440723)]
```

tf-idf 模型再次返回一个元组列表，其中第一个条目是令牌 ID，第二个条目是 tf-idf 权重。请注意，对应于 "system"（在原始语料库中出现 4 次）的 ID 的权重低于对应于 "minors"（仅出现两次）的 ID。

您可以将训练后的模型保存到磁盘，然后稍后将它们加载回来，以继续在新训练文档上进行训练或转换新文档。

`gensim` 提供了许多不同的模型/转换。有关详细信息，请参阅 [Topics and Transformations](https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#sphx-glr-auto-examples-core-run-topics-and-transformations-py)。

创建模型后，您可以使用它执行各种酷炫的操作。例如，通过 TfIdf 转换整个语料库并进行索引，以便进行相似性查询：

```python
from gensim import similarities

index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
```

以及查询我们的查询文档 `query_document` 与语料库中的每个文档的相似性：

```python
query_document = 'system engineering'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))
```

输出：

```python
[(0, 0.0), (1, 0.32448703), (2, 0.41707572), (3, 0.7184812), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]
```

如何阅读此输出？文档 3 的相似度得分是 0.718=72%，文档 2 的相似度得分是 42% 等。我们可以通过排序使其更易于阅读：

```python
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)
```

输出：

```python
3 0.7184812
2 0.41707572
1 0.32448703
0 0.0
4 0.0
5 0.0
6 0.0
7 0.0
8 0.0
```

## Summary

`gensim` 的核心概念是：

1. Document：一些文本。
2. Corpus：文档集合。
3.  Vector：文档的数学方便表示形式。
4. Model：一种用于将向量从一种表示形式转换为另一种表示形式的算法。

我们看到了这些概念的实际操作。首先，我们从一个文档语料库开始。接下来，我们将这些文档转换为向量空间表示。然后，我们创建了一个模型，将我们原始的向量表示转换为 TfIdf。最后，我们使用我们的模型来计算查询文档与语料库中所有文档之间的相似度。

## What Next?

关于[语料库和向量空间](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py)，还有很多要学习的知识。

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('run_core_concepts.png')
imgplot = plt.imshow(img)
_ = plt.axis('off')
```



Next - [Corpora and Vector Spaces](<./Corpora and Vector Spaces.md>)