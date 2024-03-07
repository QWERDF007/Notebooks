# [Signals & Slots](https://doc.qt.io/qt-6/signalsandslots.html)

信号和槽（Signals and slots）是用于对象间通信的一种机制。它是 Qt 的一个核心特性，也是与其他框架最不同的部分。信号和槽是由 Qt 的[元对象系统](<./The Meta-Object System.md>)实现的。

## Introduction

在 GUI 编程中，当我们更改一个组件时，我们通常希望另一个组件能够收到通知。更一般地，我们希望任何类型的对象都能够相互通信。例如，如果用户单击 "关闭" 按钮，则可能希望调用窗口的 `close()` 函数。

其他工具包使用回调函数实现此类通信。回调函数是指向函数的指针，因此，如果您希望处理函数通知您某个事件，则将指向另一个函数（回调）的指针传递给处理函数。然后，处理函数在适当时调用回调。虽然使用此方法的成功框架确实存在，但回调可能不直观，并可能在确保回调参数类型正确性方面存在问题。

## Signals and Slots

在 Qt 中，我们有一种替代回调技术的方法：我们使用信号和槽。当发生特定事件时会发出信号。Qt 的组件具有许多预定义信号，但我们也可以子类化组件以添加我们自己的信号。槽是响应特定信号而调用的函数。Qt 的组件具有许多预定义槽，但通常惯例是子类化组件并添加自己的槽，以便您可以处理您感兴趣的信号。

<img src="./assets/abstract-connections.png"/>

信号和槽机制是类型安全的：信号的签名必须与接收槽的签名匹配。 （实际上，槽可以比其接收到的信号具有更短的签名，因为它可以忽略额外参数）。由于签名是兼容的，因此编译器可以在使用基于函数指针的语法时帮助我们检测类型不匹配。基于字符串的 `SIGNAL` 和 `SLOT` 语法将在运行时检测到类型不匹配。信号和槽是松散耦合的：发出信号的类既不知道也不关心哪些槽接收该信号。 Qt 的信号和槽机制确保如果将信号连接到槽，则在正确时间将使用该信号参数调用该槽。 信号和槽可以带任何类型和数量的参数。 它们是完全类型安全的。

所有继承自 `QObject` 或它的子类 (如 `QWidget`) 的类都可以包含信号和槽。 当对象以一种可能会引起其他对象兴趣地方式更改其状态时，它们会发出信号。 这就是对象所做的全部通信工作。 它不知道也不关心是否有任何东西正在接收它所发射的信号。 这是真正意义上的信息封装，并确保该对象可用作软件组件。

槽可用于接收信号，但它们也是普通成员函数。就像一个对象不知道是否有任何东西接收它的信号一样，一个槽也不知道是否有任何信号连接到它。这确保了可以用 Qt 来创建真正独立的组件。

你可以将任意数量的信号连接到一个槽，并且一个信号可以根据需要连接到任意数量的槽。甚至可以直接将一个信号连接到另一个信号。(这会在第一个信号被发射时立即发射第二个信号。)

总的来说，信号和槽构成了一个强大的组件编程机制。

## Signals

信号是由对象在其内部状态以某种可能对对象的客户端或拥有者感兴趣的方式发生改变时发出的。信号是公共访问函数，可以从任何地方发出，但我们建议仅从定义信号及其子类的类中发出它们。

当信号被发射时，连接到它的槽通常会立即执行，就像普通函数调用一样。在这种情况下，信号和槽机制完全独立于任何GUI事件循环。在 `emit` 语句之后的代码将在所有槽返回后才被执行。当使用队列连接时，情况会略有不同；在这种情况下，`emit` 关键字之后的代码将立即继续，而槽将在后面执行。

如果多个槽连接到一个信号，当信号被发射时，这些槽会一个接一个按连接顺序执行。

信号由 `moc` 自动生成，不得在 .cpp 文件中实现。信号永远不能有返回类型 (即使用 `void`)。

关于参数的注意事项：我们的经验表明，如果信号和槽不使用特殊类型，它们会更具可重用性。如果 `QScrollBar::valueChanged()` 使用如假想的 `QScrollBar::Range` 这样的特殊类型，它只能连接到专门为 `QScrollBar` 设计的槽。将不同的输入组件连接在一起将是不可能的。

## Slots

槽是在连接到它的信号被发射时调用的。槽是普通的C++函数，可以正常调用；它们的唯一特殊功能是信号可以连接到它们。

由于插槽是普通成员函数，因此在直接调用时遵循普通的 C++ 规则。但是，作为插槽，它们可以通过信号-插槽连接由任何组件调用，而不考虑其访问级别。这意味着从任意类的实例发出的信号可能会导致在不相关类的实例中调用私有插槽。

您也可以将槽定义为虚函数，我们发现这在实践中非常有用。

与回调相比，信号和槽略微更慢，因为它们提供的灵活性更高，尽管对于实际应用程序来说，区别是微不足道的。通常，使用非虚拟函数调用，发出连接到某些插槽的信号大约比直接调用接收器要慢十倍。这是定位连接对象，安全迭代所有连接 (即检查后续接收者在发射期间是否被销毁) 以及以泛型方式编组任何参数所需的开销。虽然十倍非虚拟函数调用可能听起来很多，但与任何 `new` 或 `delete` 操作相比，它开销要少得多。一旦执行需要 `new` 或 `delete` 字符串、向量或列表的操作，信号和插槽开销就只占完整函数调用成本的一小部分。当您在槽中进行系统调用或间接调用超过十个函数时，情况也是如此。信号和插槽机制的简单性和灵活性完全值得这种开销，而您的用户甚至不会注意到它。

请注意，如果其他库定义了名为 `signals` 或 `slots` 的变量，则在与基于 Qt 的应用程序一起编译时可能会导致编译器警告和错误。要解决此问题，请使用 `#undef`

## A Small Example

最小的 C++ 类声明可能是这样的：

```c++
class Counter
{
public:
    Counter() { m_value = 0; }

    int value() const { return m_value; }
    void setValue(int value);

private:
    int m_value;
};
```

一个小的基于 `QObject` 类声明可能是这样：

```c++
#include <QObject>

class Counter : public QObject
{
    Q_OBJECT

public:
    Counter() { m_value = 0; }

    int value() const { return m_value; }

public slots:
    void setValue(int value);

signals:
    void valueChanged(int newValue);

private:
    int m_value;
};
```

基于 `QObject` 的版本具有相同的内部状态，并提供访问状态的公共方法，但此外它还支持使用信号和槽进行组件编程。这个类可以通过发射信号 `valueChanged()` 告诉外部世界它的状态已经改变，它还有一个槽，其他对象可以向其发送信号。

所有包含信号或槽的类必须在声明的顶部提到 `Q_OBJECT`。它们还必须 (直接或间接) 派生自 `QObject`。

槽由应用程序开发人员实现。下面是 `Counter::setValue()` 槽的一种可能实现:

```c++
void Counter::setValue(int value)
{
    if (value != m_value) {
        m_value = value;
        emit valueChanged(value);
    }
}
```

`emit` 这行代码会从该对象发出 `valueChanged()` 信号，以新的值作为参数。

在下面的代码片段中，我们创建了两个 `Counter` 对象，并使用 `QObject::connect()` 将第一个对象的 `valueChanged()` 信号连接到第二个对象的 `setValue()` 槽:

```c++
	Counter a, b;
    QObject::connect(&a, &Counter::valueChanged,
                     &b, &Counter::setValue);

    a.setValue(12);     // a.value() == 12, b.value() == 12
    b.setValue(48);     // a.value() == 12, b.value() == 48
```

调用 `a.setValue(12)` 会使 `a` 发出 `valueChanged(12)` 信号，`b` 会在它的 `setValue()` 槽中接收到这个信号，即调用 `b.setValue(12)`。然后 `b` 发出同样的 `valueChanged()` 信号，但由于没有槽连接到 `b` 的 `valueChanged()` 信号，这个信号被忽略了。

注意，`setValue()` 函数只在 `value != m_value` 时设置值并发出信号。这可以避免循环连接的无限循环 (例如，如果 `b.valueChanged()` 连接到 `a.setValue()`)。

默认情况下，**对于您创建的每个连接，都会发射一个信号；对于重复的连接会发射两个信号**。您可以使用一个 `disconnect()` 调用断开所有这些连接。如果传递 `Qt::UniqueConnection` 类型，则只在连接不重复时才建立连接。如果已经存在重复项 (完全相同的信号连接到同一对象上的完全相同的槽)，则连接将失败，`connect` 将返回 `false`。

这个例子说明对象可以在不需要了解对方任何信息的情况下协同工作。为此，对象只需要连接在一起，这可以通过一些简单的 `QObject::connect()` 函数调用或 [uic](https://doc.qt.io/qt-6/uic.html) 的 [自动连接](https://doc.qt.io/qt-6/designer-using-a-ui-file.html#automatic-connections) 功能来实现。

## A Real Example

下面是一个简单的无成员函数的窗口类头文件的示例。目的是展示如何在自己的应用程序中使用信号和槽。

```c++
#ifndef LCDNUMBER_H
#define LCDNUMBER_H

#include <QFrame>

class LcdNumber : public QFrame
{
    Q_OBJECT
```

`LcdNumber` 通过 `QFrame` 和 `QWidget` 继承自 `QObject`，后者包含了大部分的信号-槽相关知识。它与内置的 `QLCDNumber` 组件有些类似。

`Q_OBJECT` 宏会被预处理器展开为声明几个由 `moc` 实现的成员函数；如果你得到类似 "对 `LcdNumber` 的 `vtable` 的未定义的引用" 的编译错误，很可能是忘了运行 `moc` 或者没有在链接命令中包含 `moc` 输出。

```c++
public:
    LcdNumber(QWidget *parent = nullptr);

signals:
    void overflow();
```

在类构造函数和 `public` 成员之后，我们声明了类的 `signals`。当被要求显示一个不可能的值时，`LcdNumber` 类会发出 `overflow()` 信号。

如果你不关心溢出，或知道不会发生溢出，可以忽略 `overflow()` 信号，即不要连接它到任何槽。

**另一方面，如果你想在数字溢出时调用两个不同的错误函数，只需要将信号连接到两个不同的槽即可。Qt 将调用两者 (按连接顺序)。**

```c++
public slots:
    void display(int num);
    void display(double num);
    void display(const QString &str);
    void setHexMode();
    void setDecMode();
    void setOctMode();
    void setBinMode();
    void setSmallDecimalPoint(bool point);
};

#endif
```

槽是用于获取其他组件状态变化信息的接收函数。如上代码所示，`LcdNumber` 使用它来设置显示的数字。由于 `display()` 是类与程序其他部分的接口的一部分，所以该槽是公有的。

几个示例程序将 `QScrollBar` 的 `valueChanged()` 信号连接到 `display()` 槽，以便 LCD 数字持续显示滚动条的值。

请注意，`display()` 被重载了；在您将信号连接到槽时，Qt 将选择适当的版本。如果使用回调，您必须自己找到 5 个不同的名称并跟踪类型。

## Signals And Slots With Default Arguments

信号和槽的签名可以包含参数，并且参数可以具有默认值。考虑 `QObject::destroyed()`：

```c++
void destroyed(QObject* = nullptr);
```

当一个 `QObject` 被删除，它就会发出这个 `QObject::destroyed()` 信号。我们想要捕获这个信号，无论我们可能在哪里有一个指向已删除 `QObject` 的悬空引用，这样我们就可以清理它。一个合适的槽签名可能是：

```c++
void objectDestroyed(QObject* obj = nullptr);
```

要将信号连接到槽，我们使用 `QObject::connect()`。连接信号和槽的方法有多种。第一种是使用函数指针：

```c++
connect(sender, &QObject::destroyed, this, &MyObject::objectDestroyed);
```

使用函数指针调用 `QObject::connect()` 有几个优点。首先，它允许编译器检查信号的参数是否与槽的参数兼容。如果需要，参数也可以由编译器隐式转换。

您也可以连接 `functor` 或 C++11 `lambda`：

```c++
connect(sender, &QObject::destroyed, this, [=](){ this->m_objects.remove(sender); });
```

在这两种情况下，我们都在调用 `connect()` 时提供了 `this` 作为上下文。上下文对象提供关于接收者应该在哪个线程中执行的信息。这很重要，因为提供上下文可以确保接收者在上下文线程中执行。

当发送者或上下文被销毁时，`lambda` 将被断开连接。您应该注意，在 functor 中使用的任何对象在信号被发射时必须还存活。

连接信号和槽的另一种方式是使用 `QObject::connect()` 和 `SIGNAL` 和 `SLOT` 宏。关于在 `SIGNAL()` 和 `SLOT()` 宏中是否包含参数，如果参数有默认值，那么传递给 `SIGNAL()` 宏的签名不能比传递给 `SLOT()` 宏的签名含有更少的参数。

下面的全部可以工作:

```c++
connect(sender, SIGNAL(destroyed(QObject*)), this, SLOT(objectDestroyed(Qbject*)));
connect(sender, SIGNAL(destroyed(QObject*)), this, SLOT(objectDestroyed()));
connect(sender, SIGNAL(destroyed()), this, SLOT(objectDestroyed()));
```

但这一个不行：

```c++
connect(sender, SIGNAL(destroyed()), this, SLOT(objectDestroyed(QObject*)));
```

因为槽会期待一个信号不会发送的 `QObject`。这个连接会报告运行时错误。

请注意，使用这个 `QObject::connect()` 重载时，信号和槽的参数不会被编译器检查。

## Advanced Signals and Slots Usage

对于可能需要信号发送者信息的情况，Qt 提供了 `QObject::sender()` 函数，它返回发送信号的对象的指针。

`lambda` 表达式是将自定义参数传递给槽的一种方便方法:

```c++
connect(action, &QAction::triggered, engine,
        [=]() { engine->processAction(action->text()); });
```

### Using Qt with 3rd Party Signals and Slots

可以将 Qt 与第三方信号/槽机制一起使用。您甚至可以在同一个项目中使用这两种机制。为此，请在 CMake 项目文件中编写以下内容：

```c++
target_compile_definitions(my_app PRIVATE QT_NO_KEYWORDS)
```

在 qmake 项目中，你需要写：

```c++
CONFIG += no_keywords
```

它告诉 Qt 不要定义 `moc` 关键字 `signals`、`slots` 和 `emit`，因为这些名称将由第三方库使用，例如 `Boost`。然后，要在 `no_keywords` 标志下继续使用 Qt 信号和槽，只需将源代码中 Qt `moc` 关键字的所有使用替换为相应的 Qt 宏 `Q_SIGNALS` (或 `Q_SIGNAL`)、`Q_SLOTS` (或 `Q_SLOT`) 和 `Q_EMIT` 即可。

### Signals and slots in Qt-based libraries

基于 Qt 的库的公共 API 应该使用关键字 `Q_SIGNALS` 和 `Q_SLOTS` 代替 `signals` 和 `slots`。否则，在一个定义了 `QT_NO_KEYWORDS` 的项目中使用这样的库就很难。

为了强制实施这种限制，库的创建者可以在构建库时设置预处理器定义 `QT_NO_SIGNALS_SLOTS_KEYWORDS`。

这个定义排除了信号和槽，而不影响是否可以在库实现中使用其他 Qt 特定关键字。



<!-- 完成标志, 看不到, 请忽略! -->
