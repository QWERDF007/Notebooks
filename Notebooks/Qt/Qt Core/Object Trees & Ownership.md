# [Object Trees & Ownership](https://doc.qt.io/qt-6/objecttrees.html)

## Overview

`QObjects` 组织成对象树。当您创建一个具有另一个对象作为父对象的 `QObject` 时，它将添加到父对象的`children()` 列表中，并在父对象被删除时删除。事实证明，这种方法非常适合 GUI 对象的需求。例如，`QShortcut` (键盘快捷方式) 是相关窗口的子项，因此当用户关闭该窗口时，快捷方式也会被删除。

`QQuickItem` 是 Qt Quick 模块的基本可视元素，继承自 `QObject`，但具有与 `QObject` 父项不同的视觉父项概念。项目的视觉父项可能与其 object 父项不同。详细信息请参见 [Concepts - Visual Parent in Qt Quick](https://doc.qt.io/qt-6/qtquick-visualcanvas-visualparent.html)。

`QWidget` 是 Qt Widgets 模块的基本类，扩展了父子关系。子项通常也成为子窗口小部件，即在其父窗口小部件的坐标系中显示，并且在图形上由其父边界剪切。例如，当应用程序在关闭消息框后删除消息框时，消息框的按钮和标签也将被删除，就像我们希望的那样，因为按钮和标签是消息框的子项。

您还可以自己删除子对象，并且它们将从其父对象中删除。例如，当用户删除工具栏时，它可能会导致应用程序删除其其中一个 `QToolBar` 对象，在这种情况下，工具栏的 `QMainWindow` 父项将检测到更改并相应地重新配置其屏幕空间。

当应用程序看起来或行为奇怪时，调试函数 [`QObject::dumpObjectTree()`](https://doc.qt.io/qt-6/qobject.html#dumpObjectTree) 和 [`QObject::dumpObjectInfo()`](https://doc.qt.io/qt-6/qobject.html#dumpObjectInfo) 通常很有用。

## Construction/Destruction Order of QObjects