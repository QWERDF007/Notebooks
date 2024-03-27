# [QML Object Types](https://doc.qt.io/qt-6/qtqml-typesystem-objecttypes.html)

QML 对象类型是用于创建 QML 对象的模板。

语法上，QML 对象类型可以用类型名称和包含对象属性的大括号对来声明对象。这不同于值类型，值类型无法以相同的方式使用。例如，`Rectangle` 是一个 QML 对象类型，可以用来创建 `Rectangle` 类型的对象。诸如 `int` 和 `bool` 等基本类型则用于保存简单数据而非对象，因此不能像对象类型那样使用。

您可以通过创建 `.qml` 文件来定义自定义的 QML 对象类型，该文件用于定义类型，具体方法请参见[使用 QML 文档定义对象类型](https://doc.qt.io/qt-6/qtqml-documents-definetypes.html)部分。或者，也可以按照[使用 C++ 定义 QML 类型](<./Defining QML Types from C++.md>)部分的介绍，通过 C++ 定义 QML 类型并将其注册到 QML 引擎中。需要注意的是，在这两种情况下，类型名称都必须以大写字母开头，才能在 QML 文件中声明为 QML 对象类型。

有关 C++ 和不同 QML 集成方法的更多信息，请参阅 [C++ 和 QML 集成概述](<./Overview - QML and C++ Integration.md>)页面。

## Defining Object Types from QML

### Defining Object Types Through QML Documents

插件编写者和应用程序开发人员可以提供通过 QML 文档定义的类型。QML 文档在 QML 导入系统可见时，会根据文件名（去除扩展名）定义一个类型。

因此，如果存在名为 "MyButton.qml" 的 QML 文档，那么它就提供了 `MyButton` 类型的定义，该类型可以在 QML 应用程序中使用。

有关如何定义 QML 文档以及 QML 语法的语法，请参阅 [QML 文档](https://doc.qt.io/qt-6/qtqml-documents-topic.html)的相关文档。熟悉了 QML 语言以及如何定义 QML 文档之后，可以查阅相关文档，了解如何在 QML 文档中[定义和使用您自己可重用的 QML 类型](https://doc.qt.io/qt-6/qtqml-documents-definetypes.html)。

有关详细信息，请参阅[使用 QML 文档定义对象类型](https://doc.qt.io/qt-6/qtqml-documents-definetypes.html)。

### Defining Anonymous Types with Component

另一种在 QML 内部创建对象类型的方法是使用 `Component` 类型。这允许在 QML 文档中内联定义类型，而不是使用单独的 `.qml` 文件。

```javascript
Item {
    id: root
    width: 500; height: 500

    Component {
        id: myComponent
        Rectangle { width: 100; height: 100; color: "red" }
    }

    Component.onCompleted: {
        myComponent.createObject(root)
        myComponent.createObject(root, {"x": 200})
    }
}
```

这里的 `myComponent` 对象本质上定义了一个匿名类型，可以使用 `Component::createObject` 来创建这种匿名类型的对象。

内联组件共享常规顶级组件的所有特性，并使用与其包含的 QML 文档相同的导入列表。

请注意，每个 `Component` 对象声明都会创建自己的组件作用域。在 `Component` 对象声明内部使用和引用的任何 `id` 值都必须在该作用域内唯一，但不必在声明内联组件的文档中唯一。因此，`myComponent` 对象声明中声明的 `Rectangle` 可以具有 `root` 的 `id`，而不会与同一文档中为 `Item` 对象声明的 `root` 冲突，因为这两个 `id` 值是在不同的组件作用域内声明的。

有关更多详细信息，请参阅[作用域和命名解析](https://doc.qt.io/qt-6/qtqml-documents-scope.html)。

## Defining Object Types from C++

C++ 插件编写者和应用程序开发人员可以通过 Qt QML 模块提供的 API 注册用 C++ 定义的类型。存在各种注册函数，每个函数可以满足不同的用例。有关这些注册函数以及将自定义 C++ 类型公开到 QML 的具体信息，请参阅[使用 C++ 定义 QML 类型](<./Defining QML Types from C++.md>)的相关文档。

QML 类型系统依赖于将导入、插件和扩展安装到已知的导入路径。插件可以由第三方开发人员提供，并供客户端应用程序开发人员重复使用。有关如何创建和部署 QML 扩展模块的更多信息，请参阅关于 [QML 模块](https://doc.qt.io/qt-6/qtqml-modules-topic.html)的文档。



<!-- 完成标志, 看不到, 请忽略! -->