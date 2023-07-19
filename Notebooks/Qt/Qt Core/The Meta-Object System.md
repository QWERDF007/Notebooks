# [The Meta-Object System](https://doc.qt.io/qt-6/metaobjects.html)

Qt 的元对象系统提供了对象间通信的信号和槽机制、运行时类型信息和动态属性系统。

元对象系统基于三个方面：

- `QObject` 类为支持元对象系统的对象提供了一个基类。
- 类声明的私有部分中的 `Q_OBJECT` 宏用于启用元对象特性，例如动态属性、信号和槽。
- [元对象编译器](https://doc.qt.io/qt-6/moc.html) (moc) 为每个 `QObject` 子类提供必要的代码来实现元对象特性。

moc 工具读取一个 C++ 源文件。如果它找到一个或多个包含 Q_OBJECT 宏的类声明，它将为这些类中的每一个类生成包含元对象代码的另一个 C++ 源文件。这个生成的源文件要么被 `#include` 到类的源文件中，要么更常见的是与类的实现一起编译和链接。

除了为对象间通信提供 [信号和槽](./Signals & Slots.md) 机制（引入该系统的主要原因）之外，元对象代码还提供了以下其他功能：

- `QObject::metaObject()` 返回与类关联的元对象。
- `QMetaObject::className()` 在运行时返回类名作为字符串，而不需要通过 C++ 编译器支持本机运行时类型信息 (RTTI)。
- `QObject::inherits()` 函数返回一个对象是否是在 `QObject` 继承树内继承了指定类的实例。
- `QObject::tr()` 用于国际化翻译字符串。
- `QObject::setProperty()` 和 `QObject::property()` 可以通过名称动态设置和获取属性。
- `QMetaObject::newInstance()` 构造类的新实例。 

还可以使用 `qobject_cast()` 在 `QObject` 类上执行动态转换。`qobject_cast()` 函数的行为类似于标准 C++ `dynamic_cast()`，具有不需要 RTTI 支持和跨动态库边界工作的优点。它尝试将其参数转换为尖括号中指定的指针类型，如果对象是正确类型 (在运行时确定)，则返回非零指针，否则返回 `nullptr`。

例如，假设 `MyWidget` 继承自 `QWidget` 并使用 `Q_OBJECT` 宏声明：

```c++
QObject *obj = new MyWidget;
```

类型为 `QObject *` 的变量 `obj` ，实际上是指向 `MyWidget` 对象的指针，因此我们可以进行适当的转换：

```c++
QWidget *widget = qobject_cast<QWidget *>(obj);
```

从 `QObject` 到 `QWidget` 的转换成功，因为对象实际上是 `MyWidget`，它是 `QWidget` 的子类。由于我们知道 `obj` 是一个 `MyWidget`，因此我们还可以将其转换为 `MyWidget *`：

```c++
MyWidget *myWidget = qobject_cast<MyWidget *>(obj);
```

转换为 `MyWidget` 成功，因为 `qobject_cast()` 不区分内置 Qt 类型和自定义类型。

```c++
QLabel *label = qobject_cast<QLabel *>(obj);
// label is 0
```

另一方面，转换为 `QLabel` 失败。然后将指针设置为 0。这使得在运行时基于类型以不同方式处理不同类型的对象成为可能：

```c++
if (QLabel *label = qobject_cast<QLabel *>(obj)) {
    label->setText(tr("Ping"));
} else if (QPushButton *button = qobject_cast<QPushButton *>(obj)) {
    button->setText(tr("Pong!"));
}
```

虽然可以在没有 `Q_OBJECT` 宏和元对象代码的情况下使用 `QObject` 作为基类，但如果不使用 `Q_OBJECT` 宏，则既没有信号和槽，也没有这里描述的其他功能可用。从元对象系统的角度来看，没有元代码的 `QObject` 子类等效于具有元对象代码的最近祖先。这意味着例如 `QMetaObject::className()` 将不返回您的类的实际名称，而是返回该祖先的类名。

因此，我们**强烈建议所有 `QObject` 的子类都使用 `Q_OBJECT` 宏**，无论它们是否实际上使用信号、槽和属性。

另请参见 [QMetaObject](https://doc.qt.io/qt-6/qmetaobject.html)、[Qt 的属性系统](./The Property System.md) 和 [信号和槽](./Signals & Slots.md)。

