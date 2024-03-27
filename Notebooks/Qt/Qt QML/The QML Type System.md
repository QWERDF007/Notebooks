# [The QML Type System](https://doc.qt.io/qt-6/qtqml-typesystem-topic.html)

QML 文档中对象层次结构定义中可能使用的类型可以来自各种来源。它们可以是：

- 由 QML 语言原生提供的
- 通过 QML 模块用 C++ 注册的
- 由 QML 模块作为 QML 文档提供的

此外，应用程序开发人员还可以提供自己的类型，既可以直接注册 C++ 类型，也可以在 QML 文档中定义可重用的组件，然后导入这些组件。

无论类型定义来自何处，引擎都会对这些类型的属性和实例强制执行类型安全性。

## Value Types

QML 语言内置支持各种基本类型，包括整数、双精度浮点数、字符串和布尔值。对象可以具有这些类型的属性，这些类型的的值可以作为参数传递给对象的函数。

有关值类型的更多信息，请参阅 [QML 值类型](<./QML Value Types.md>)文档。

## JavaScript Types

QML 引擎支持 JavaScript 对象和数组。可以使用通用类型 [var](https://doc.qt.io/qt-6/qml-var.html) 创建和存储任何标准的 JavaScript 类型。

例如，标准的 `Date` 和 `Array` 类型可用，如下所示：

```javascript
import QtQuick 2.0

Item {
    property var theArray: []
    property var theDate: new Date()

    Component.onCompleted: {
        for (var i = 0; i < 10; i++)
            theArray.push("Item " + i)
        console.log("There are", theArray.length, "items in the array")
        console.log("The time is", theDate.toUTCString())
    }
}
```

有关详细信息，请参阅 [QML 文档中的 JavaScript 表达式](https://doc.qt.io/qt-6/qtqml-javascript-expressions.html)。

## QML Object Types

QML 对象类型是从该类型可以实例化 QML 对象的类型。QML 对象类型派生自 [QtObject](https://doc.qt.io/qt-6/qml-qtqml-qtobject.html)，由 QML 模块提供。应用程序可以导入这些模块来使用它们提供的对象类型。QtQuick 模块提供了用于在 QML 中创建用户界面的最常用对象类型。

最后，每个 QML 文档隐式地定义了一个 QML 对象类型，该类型可以在其他 QML 文档中重复使用。有关对象类型的详细信息，请参阅 [QML 类型系统中的对象类型](<./QML Object Types.md>)文档。



<!-- 完成标志, 看不到, 请忽略! -->