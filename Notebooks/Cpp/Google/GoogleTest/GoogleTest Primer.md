# [GoogleTest Primer](https://google.github.io/googletest/primer.html)

## 介绍：为什么选择 GoogleTest？

GoogleTest 帮助你编写更好的 C++ 测试。

GoogleTest 是一个由测试技术团队开发的测试框架，它考虑到了 Google 的具体需求和限制。无论你是在 Linux、Windows 还是 Mac 上工作，如果你编写 C++ 代码，GoogleTest 都可以帮助你。它不仅支持单元测试，还支持任何类型的测试。

那么，什么样的测试是好的，GoogleTest 又是如何适应的呢？我们相信：

1. 测试应该是独立且可重复的。调试一个因为其他测试而成功或失败的测试是痛苦的。GoogleTest 通过在不同的对象上运行每个测试来隔离测试。当测试失败时，GoogleTest 允许你将其隔离运行，以便快速调试。

2. 测试应该组织良好，并反映被测试代码的结构。GoogleTest 将相关的测试分组到测试套件中，这些测试套件可以共享数据和子程序。这种模式很容易识别，使测试易于维护。当人员切换项目并开始在一个新的代码库上工作时，这种一致性尤其有帮助。

3. 测试应该是可移植和可重用的。Google 有很多与平台无关的代码；它的测试也应该是平台无关的。GoogleTest 在不同的操作系统上工作，使用不同的编译器，有或没有异常，因此 GoogleTest 测试可以与各种配置一起工作。

4. 当测试失败时，它们应该提供尽可能多的问题信息。GoogleTest 不会停在第一个测试失败上。相反，它只停止当前测试并继续下一个。你也可以设置测试，在报告非致命失败后当前测试继续进行。因此，你可以在一次运行-编辑-编译周期中检测并修复多个错误。

5. 测试框架应该使测试编写者摆脱家务琐事，让他们专注于测试内容。GoogleTest 自动跟踪所有定义的测试，并且不需要用户枚举它们才能运行它们。

6. 测试应该是快速的。使用 GoogleTest，你可以跨测试重用共享资源，并且只需要支付一次设置/拆卸费用，而不需要使测试相互依赖。

由于 GoogleTest 基于流行的 xUnit 架构，如果你以前使用过 JUnit 或 PyUnit，你会感到宾至如归。如果没有，学习基础知识并开始使用大约需要 10 分钟。那么，让我们开始吧！

## 注意术语的使用

> 注意：由于*测试*、*测试用例*和*测试套件*这些术语的不同定义可能会引起一些混淆，所以要小心误解。

从历史上看，GoogleTest 开始使用*测试用例* (Test Case) 这个术语来分组相关的测试，而目前的出版物，包括国际软件测试资格委员会（ISTQB）的材料和各种软件质量教科书，都使用*测试套件* ([Test Suite](https://glossary.istqb.org/en_US/term/test-suite-1-3)) 这个术语。

在 GoogleTest 中使用的相关术语*测试*，对应于 ISTQB 和其他地方的*测试用例*。

*测试*这个术语通常足够宽泛，包括 ISTQB 对*测试用例*的定义，所以这里问题不大。但是，GoogleTest 中使用的*测试用例*这个术语是矛盾的，因此令人困惑。

GoogleTest 最近开始用*测试套件*替换*测试用例*这个术语。首选的 API 是 TestSuite。旧的 TestCase API 正在被缓慢弃用和重构。

因此，请注意这些术语的不同定义：

| 含义                                         | GoogleTest 术语       | ISTQB 术语 |
| -------------------------------------------- | --------------------- | ---------- |
| 使用特定的输入值执行特定的程序路径并验证结果 | [TEST()](#<简单测试>) | Test Case  |

## 基本概念

使用 GoogleTest 时，你从编写断言开始，它是检查条件是否为真的语句。断言的结果可以是成功、非致命失败或致命失败。如果出现致命失败，它会中止当前函数；否则程序正常继续。

测试使用断言来验证被测试代码的行为。如果测试崩溃或断言失败，则测试失败；否则测试成功。

一个测试套件包含一个或多个测试。你应该将测试分组到反映被测试代码结构的测试套件中。当测试套件中的多个测试需要共享公共对象和子程序时，你可以将它们放入测试夹具类 (*test fixture*) 中。

一个测试程序可以包含多个测试套件。

我们现在将解释如何编写测试程序，从单个断言级别开始，逐步构建到测试和测试套件。

## 断言

GoogleTest 断言是类似于函数调用的宏。通过对其行为做出断言来测试一个类或函数。当断言失败时，GoogleTest 会打印断言的源文件和行号位置，以及失败消息。你也可以提供一个自定义的失败消息，该消息将被附加到 GoogleTest 的消息中。

这些断言成对出现，测试相同的东西，但对当前函数有不同的影响。当失败时，`ASSERT_*` 版本会生成致命失败，并中止当前函数。`EXPECT_*` 版本生成非致命失败，不会中止当前函数。通常首选 `EXPECT_*`，因为它们允许在测试中报告多个失败。然而，如果在相关断言失败时继续没有意义，你应该使用 `ASSERT_*`。

由于失败的 `ASSERT_*` 会立即从当前函数返回，可能会跳过它之后的清理代码，因此可能会导致内存泄漏。根据泄漏的性质，它可能值得修复也可能不值得修复——所以如果你除了断言错误之外还得到了堆检查器错误，请记住这一点。

要提供自定义的失败消息，只需使用 `<<` 操作符将其流式传输到宏中。参见以下示例，使用 ASSERT_EQ 和 EXPECT_EQ 宏来验证值相等性：

```c++
ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";

for (int i = 0; i < x.size(); ++i) {
  EXPECT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;
}
```


任何可以流到 ostream 的内容都可以流到断言宏中——特别是，C 字符串和 `string` 对象。如果将宽字符串（在 Windows 上的 `UNICODE` 模式下为 `wchar_t*`，`TCHAR*` 或 `std::wstring`）流到断言中，打印时会被转换为 UTF-8。

GoogleTest 提供了一系列断言，以多种方式验证你的代码的行为。你可以检查布尔条件，基于关系运算符比较值，验证字符串值、浮点值等等。甚至还有断言可以通过提供自定义谓词来启用你验证更复杂的状态。有关 GoogleTest 提供的断言的完整列表，请参阅断言参考。

## 简单测试

要创建一个测试：

1. 使用 `TEST()` 宏来定义和命名一个测试函数。这些是普通的不返回值的 C++ 函数。
2. 在这个函数中，除了你想包含的任何有效的 C++ 语句外，使用各种 GoogleTest 断言来检查值。
3. 测试的结果由断言决定；如果测试中的任何断言失败（无论是致命的还是非致命的），或者测试崩溃，整个测试就失败了。否则，它就成功了。

```c++
TEST(TestSuiteName, TestName) {
  ... test body ...
}
```

`TEST()` 的参数从一般到具体。第一个参数是测试套件 (test suite) 的名字，第二个参数是测试套件内的测试名字。两个名字都必须是有效的 C++ 标识符，并且它们不应该包含任何下划线（`_`）。一个测试的全名由它的包含测试套件和它的测试名字组成。来自不同测试套件的测试可以具有相同的测试名字。

例如，让我们来看一个简单的整数函数：

```c++
int Factorial(int n);  // 返回 n 的阶乘
```

这个函数的测试套件可能看起来像这样：

```c++
// 测试 0 的阶乘。
TEST(FactorialTest, HandlesZeroInput) {
  EXPECT_EQ(Factorial(0), 1);
}

// 测试正数的阶乘。
TEST(FactorialTest, HandlesPositiveInput) {
  EXPECT_EQ(Factorial(1), 1);
  EXPECT_EQ(Factorial(2), 2);
  EXPECT_EQ(Factorial(3), 6);
  EXPECT_EQ(Factorial(8), 40320);
}
```


GoogleTest 通过测试套件对测试结果进行分组，因此逻辑相关的测试应该在同一个测试套件中；换句话说，它们的 `TEST()` 的第一个参数应该是相同的。在上面的例子中，我们有两个测试，`HandlesZeroInput` 和 `HandlesPositiveInput`，它们属于同一个测试套件 `FactorialTest`。

在命名你的测试套件和测试时，你应该遵循与[命名函数和类](https://google.github.io/styleguide/cppguide.html#Function_Names)相同的约定。

可用性：Linux、Windows、Mac。

## 测试夹具 (Test Fixtures)：为多个测试使用相同的数据配置
如果你发现自己正在编写两个或更多的测试，它们操作相似的数据，你可以使用测试夹具。这允许你为几个不同的测试重用相同的对象配置。

要创建一个夹具：

1. 从 `testing::Test` 派生一个类。以 `protected:` 开始其主体，因为我们希望从子类访问夹具成员。
2. 在类内部，声明你计划使用的任何对象。
3. 如果有必要，编写一个默认构造函数或 `SetUp()` 函数来为每个测试准备对象。一个常见的错误是使用小写的 u 将 `SetUp()` 拼写为 `Setup()`，使用 C++11 中的 override 确保你拼写正确。
4. 如果有必要，编写一个析构函数或 `TearDown()` 函数来释放你在 `SetUp()` 中分配的任何资源。要了解何时应该使用构造函数/析构函数，何时应该使用 `SetUp()/TearDown()`，请阅读 [FAQ](https://google.github.io/googletest/faq.html#CtorVsSetUp)。
5. 如果需要，为你的测试定义子程序以共享。
   

使用夹具时，使用 `TEST_F()` 而不是 `TEST()`，因为它允许你访问测试夹具中的对象和子程序：

```c++
TEST_F(TestFixtureClassName, TestName) {
  ... test body ...
}
```

与 `TEST()` 不同，在 `TEST_F()` 中，第一个参数必须是测试夹具类的名称。(`_F` 代表“Fixture”)。这个宏没有指定测试套件名称。

不幸的是，C++ 宏系统不允许我们创建一个单一的宏来处理两种类型的测试。使用错误的宏会导致编译器错误。

此外，你首先必须先定义一个测试夹具类，然后才能在 `TEST_F()` 中使用它，否则你会得到编译器错误 `virtual outside class declaration`。

对于每个使用 `TEST_F()` 定义的测试，GoogleTest 在运行时都会创建一个新的测试夹具，立即通过 `SetUp()` 初始化它，运行测试，通过调用 `TearDown()` 清理，然后删除测试夹具。请注意，同一测试套件中的不同测试有不同的测试夹具对象，GoogleTest 总是在创建下一个夹具之前删除当前的夹具。GoogleTest 不会为多个测试重用同一个测试夹具。一个测试对夹具所做的任何更改都不会影响其他测试。

作为一个例子，让我们为名为 `Queue` 的 FIFO 队列类编写测试，它具有以下接口：

```c++
template <typename E>  // E 是元素类型。
class Queue {
 public:
  Queue();
  void Enqueue(const E& element);
  E* Dequeue();  // 如果队列为空，则返回 NULL。
  size_t size() const;
  ...
};
```

首先，定义一个夹具类。按照惯例，你应该给它命名为 `FooTest`，其中 `Foo` 是被测试的类。

```c++
class QueueTest : public testing::Test {
 protected:
  QueueTest() {
     // q0_ 保持为空
     q1_.Enqueue(1);
     q2_.Enqueue(2);
     q2_.Enqueue(3);
  }

  // ~QueueTest() override = default;

  Queue<int> q0_;
  Queue<int> q1_;
  Queue<int> q2_;
};
```

在这种情况下，我们不需要定义析构函数或 `TearDown()` 方法，因为编译器生成的隐式析构函数将执行所有必要的清理。

现在我们使用 `TEST_F()` 和这个夹具来编写测试。

```c++
TEST_F(QueueTest, IsEmptyInitially) {
  EXPECT_EQ(q0_.size(), 0);
}

TEST_F(QueueTest, DequeueWorks) {
  int* n = q0_.Dequeue();
  EXPECT_EQ(n, nullptr);

  n = q1_.Dequeue();
  ASSERT_NE(n, nullptr);
  EXPECT_EQ(*n, 1);
  EXPECT_EQ(q1_.size(), 0);
  delete n;

  n = q2_.Dequeue();
  ASSERT_NE(n, nullptr);
  EXPECT_EQ(*n, 2);
  EXPECT_EQ(q2_.size(), 1);
  delete n;
}
```

以上使用了 `ASSERT_*` 和 `EXPECT_*` 断言。规则是，当你希望在断言失败后继续测试以揭示更多错误时，使用 `EXPECT_*`，而在失败后继续没有意义时，使用 `ASSERT_*`。例如，`Dequeue` 测试中的第二个断言是 `ASSERT_NE(n, nullptr)`，因为我们需要在之后解引用指针 `n`，当 `n` 为 `NULL` 时会导致段错误。

当这些测试运行时，会发生以下情况：

1. GoogleTest 构造一个 `QueueTest` 对象（我们称之为 `t1`）。

2. 第一个测试（`IsEmptyInitially`）在 `t1` 上运行。

3. `t1` 被析构。

4. 上述步骤在另一个 `QueueTest` 对象上重复，这次运行 `DequeueWorks` 测试。

可用性：Linux、Windows、Mac。

## 调用测试

`TEST()` 和 `TEST_F()` 隐式地将它们的测试注册到 GoogleTest 中。因此，与许多其他 C++ 测试框架不同，您不必重新列出所有定义的测试就能运行它们。

在定义了您的测试之后，您可以使用 `RUN_ALL_TESTS()` 来运行它们，如果所有测试都成功，则返回 0，否则返回 1。请注意，`RUN_ALL_TESTS()` 运行您链接单元中的所有测试——它们可以来自不同的测试套件，甚至不同的源文件。

当被调用时，`RUN_ALL_TESTS()` 宏：

- 保存所有 GoogleTest 标志的状态。
- 为第一个测试创建一个测试夹具对象。
- 通过 `SetUp()` 初始化它。
- 在夹具对象上运行测试。
- 通过 `TearDown()` 清理夹具。
- 删除夹具。
- 恢复所有 GoogleTest 标志的状态。
- 重复上述步骤进行下一个测试，直到所有测试都已运行。

如果发生致命失败，将跳过后续步骤。

> 重要提示：您不能忽略 `RUN_ALL_TESTS()` 的返回值，否则将得到编译器错误。这种设计的原因是自动化测试服务根据其退出代码来判断测试是否通过，而不是根据其 stdout/stderr 输出；因此，您的 `main()` 函数必须返回 `RUN_ALL_TESTS()` 的值。
>
> 另外，您应该只调用一次 `RUN_ALL_TESTS()`。多次调用它与一些高级 GoogleTest 特性（例如，线程安全的死亡测试）冲突，因此不支持。

可用性：Linux、Windows、Mac。

## 编写 main() 函数
大多数用户不需要编写自己的 `main` 函数，而是应该与 `gtest_main` 链接（而不是与 `gtest` 链接），它定义了一个合适的入口点。有关详情，请参见本节末尾。本节的其余部分仅在您需要在测试运行之前执行一些无法在夹具和测试套件框架内表达的自定义操作时适用。

如果您编写自己的 `main` 函数，它应该返回 `RUN_ALL_TESTS()` 的值。

您可以从以下样板代码开始：

```c++
#include "this/package/foo.h"
#include <gtest/gtest.h>

namespace my {
namespace project {
namespace {

// 测试类 Foo 的夹具。
class FooTest : public testing::Test {
 protected:
  // 如果它们的主体将是空的，您可以删除以下任何或所有函数。

  FooTest() {
     // 您可以在这里为每个测试做设置工作。
  }

  ~FooTest() override {
     // 您可以在这里做不抛出异常的清理工作。
  }

  // 如果构造函数和析构函数不足以设置和清理每个测试，您可以定义以下方法：

  void SetUp() override {
     // 这里的代码将在构造函数之后立即被调用（每个测试之前）。
  }

  void TearDown() override {
     // 这里的代码将在每个测试之后立即被调用（析构函数之前）。
  }

  // 在这里声明的类成员可以被测试套件中的所有测试用于 Foo。
};

// 测试 Foo::Bar() 方法是否执行了 Abc。
TEST_F(FooTest, MethodBarDoesAbc) {
  const std::string input_filepath = "this/package/testdata/myinputfile.dat";
  const std::string output_filepath = "this/package/testdata/myoutputfile.dat";
  Foo f;
  EXPECT_EQ(f.Bar(input_filepath, output_filepath), 0);
}

// 测试 Foo 是否执行了 Xyz。
TEST_F(FooTest, DoesXyz) {
  // 测试 Foo 的 Xyz 特性。
}

}  // namespace
}  // namespace project
}  // namespace my

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```
`testing::InitGoogleTest()` 函数解析命令行以获取 GoogleTest 标志，并移除所有已识别的标志。这允许用户通过各种标志控制测试程序的行为，我们将在高级指南中介绍。您必须在调用 `RUN_ALL_TESTS()` 之前调用此函数，否则标志不会被正确初始化。

在 Windows 上，`InitGoogleTest()` 也适用于宽字符串，因此也可以在 `UNICODE` 模式下编译的程序中使用。

但也许你认为编写所有这些 `main` 函数工作量太大？我们完全同意你的看法，这就是为什么 Google Test 提供了一个 main() 的基本实现。如果它符合您的需求，那么只需将您的测试与 `gtest_main` 库链接，您就可以开始了。

> 注意：`ParseGUnitFlags()` 已被 `InitGoogleTest()` 取代。

## 已知限制
Google Test 被设计为线程安全。在可用 `pthreads` 库的系统上，实现是线程安全的。目前在其他系统（例如 Windows）上，同时从两个线程使用 Google Test 断言是不安全的。在大多数测试中，这不是问题，因为断言通常在主线程中完成。如果您想帮忙，可以自告奋勇在 `gtest-port.h` 中为您的平台实现必要的同步原语。