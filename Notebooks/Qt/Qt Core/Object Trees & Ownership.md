# [Object Trees & Ownership](https://doc.qt.io/qt-6/objecttrees.html)

## Overview

`QObjects` 组织成对象树。当您创建一个具有另一个对象作为父对象的 `QObject` 时，它将添加到父对象的`children()` 列表中，并在父对象被删除时删除。事实证明，这种方法非常适合 GUI 对象的需求。例如，`QShortcut` (键盘快捷方式) 是相关窗口的子项，因此当用户关闭该窗口时，快捷方式也会被删除。

`QQuickItem` 是 Qt Quick 模块的基本可视元素，继承自 `QObject`，但具有与 `QObject` 父项不同的视觉父项概念。项目的视觉父项可能与其 object 父项不同。详细信息请参见 [Concepts - Visual Parent in Qt Quick](https://doc.qt.io/qt-6/qtquick-visualcanvas-visualparent.html)。

`QWidget` 是 Qt Widgets 模块的基本类，扩展了父子关系。子项通常也成为子窗口小部件，即在其父窗口小部件的坐标系中显示，并且在图形上由其父边界剪切。例如，当应用程序在关闭消息框后删除消息框时，消息框的按钮和标签也将被删除，就像我们希望的那样，因为按钮和标签是消息框的子项。

您还可以自己删除子对象，并且它们将从其父对象中删除。例如，当用户删除工具栏时，它可能会导致应用程序删除其其中一个 `QToolBar` 对象，在这种情况下，工具栏的 `QMainWindow` 父项将检测到更改并相应地重新配置其屏幕空间。

当应用程序看起来或行为奇怪时，调试函数 [`QObject::dumpObjectTree()`](https://doc.qt.io/qt-6/qobject.html#dumpObjectTree) 和 [`QObject::dumpObjectInfo()`](https://doc.qt.io/qt-6/qobject.html#dumpObjectInfo) 通常很有用。

## Construction/Destruction Order of QObjects

当 `QObjects` 在堆上创建时（即使用 `new` 创建时），可以从它们中的任何顺序构建树，稍后，树中的对象可以以任何顺序销毁。当树中的任何 `QObject` 被删除时，如果该对象有父项，则析构函数会自动从其父项中删除该对象。如果该对象有子项，则析构函数会自动删除每个子项。无论销毁的顺序如何，都不会删除两次 `QObject`。

当 `QObjects` 在栈上创建时，相同的行为适用。通常，销毁的顺序仍然不会出现问题。请考虑以下代码片段：

```c++
int main()
{
    QWidget window;
    QPushButton quit("Quit", &window);
    ...
}
```

父项 `window` 和子项 `quit` 都是 `QObjects`，因为 `QPushButton` 继承自 `QWidget`，`QWidget` 继承自 `QObject`。这段代码是正确的：`quit` 的析构函数不会被调用两次，因为C++语言标准 (ISO/IEC 14882:2003) 规定，局部对象的析构函数按其构造函数的相反顺序调用。因此，在调用 `window` 的析构函数之前，首先调用了子项 `quit` 的析构函数，并从其父项 `window` 中删除了它。

但是，现在考虑一下如果我们交换构造顺序会发生什么，如下面的代码片段所示：

```c++
int main()
{
    QPushButton quit("Quit");
    QWidget window;

    quit.setParent(&window);
    ...
}
```

在这种情况下，销毁的顺序会导致问题。因为父类是最后创建的，所以它的析构函数首先被调用。然后它调用了它的子项 `quit` 的析构函数，这是不正确的，因为`quit` 是一个局部变量。当 `quit` 随后超出范围时，它的析构函数再次被调用，这次是正确的，但已经造成了损坏。





<!-- 完成标志, 看不到, 请忽略! -->
