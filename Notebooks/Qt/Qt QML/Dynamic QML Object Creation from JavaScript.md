# [Dynamic QML Object Creation from JavaScript](https://doc.qt.io/qt-6/qtqml-javascript-dynamicobjectcreation.html)

QML 支持从 JavaScript 内部动态创建对象。这对于延迟对象实例化到需要时非常有用，从而可以改善应用程序启动时间。它还允许根据用户输入或其他事件动态创建可视对象并将其添加到场景中。

有关本页讨论的概念的演示，请参见[动态场景示例](https://doc.qt.io/qt-6/qtqml-dynamicscene-example.html)。

## Creating Objects Dynamically

从 JavaScript 动态创建对象有两种方法。您可以调用 `Qt.createComponent()` 来动态创建 `Component` 对象，或者使用 `Qt.createQmlObject()` 从 QML 字符串创建对象。如果您已经拥有在 QML 文档中定义的组件并且想要动态创建该组件的实例，那么创建组件会更好。否则，当对象的 QML 本身是在运行时生成的，从 QML 字符串创建对象会很有用。

### Creating a Component Dynamically

要动态加载在 QML 文件中定义的组件，请在 Qt 对象中调用 `Qt.createComponent()` 函数。此函数仅接受 QML 文件的 URL 作为参数，并从此 URL 创建一个 `Component` 对象。

创建 `Component` 对象后，您可以调用其 `createObject()` 方法来创建组件的实例。此函数可以接受一个或两个参数：

- 第一个是新对象的 `parent`。`parent` 可以是图形对象（即 [`Item`](https://doc.qt.io/qt-6/qml-qtquick-item.html) 类型）或非图形对象（即 `QtObject` 或 C++ `QObject` 类型）。只有具有图形父对象的图形对象才会被渲染到 Qt Quick 可视画布上。如果您想稍后设置父级，可以安全地将 `null` 传递给此函数。

- 第二个参数是可选的，它是一个属性—值对的映射，用于定义对象的任何属性值的初始值。此参数指定的属性值会在对象创建完成之前应用于对象，从而避免可能因某些属性必须初始化才能启用其他属性绑定而导致的绑定错误。此外，与在创建对象后定义属性值和绑定相比，这种方法还有一些小的性能优势。

这是一个例子。首先是 Sprite.qml，它定义了一个简单的 QML 组件：

```javascript
import QtQuick

Rectangle { width: 80; height: 50; color: "red" }
```

我们的主程序文件 `main.qml` 导入一个将会创建 `Sprite` 对象的 `componentCreation.js` 文件：

```javascript
import QtQuick
import "componentCreation.js" as MyScript

Rectangle {
    id: appWindow
    width: 300; height: 300

    Component.onCompleted: MyScript.createSpriteObjects();
}
```

