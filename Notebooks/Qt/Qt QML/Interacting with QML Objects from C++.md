# [Interacting with QML Objects from C++](https://doc.qt.io/qt-6/qtqml-cppintegration-interactqmlfromcpp.html)

所有 QML 对象类型，无论是由引擎内部实现还是由第三方源码定义，都继承自 `QObject` 类。这意味着 QML 引擎可以使用 [Qt 元对象系统](<../Qt Core/The Meta-Object System.md>)来动态实例化任何 QML 对象类型并检查创建的对象。

这对于从 C++ 代码创建 QML 对象非常有用，可以用于显示可视化呈现的 QML 对象，或者将非可视的 QML 对象数据集成到 C++ 应用中。创建 QML 对象后，可以从 C++ 检查它以读写属性、调用方法和接收信号通知。

有关 C++ 和不同 QML 集成方法的详细信息，请参阅 [C++ 和 QML 集成概述](<./Overview - QML and C++ Integration.md>)页面。

## Loading QML Objects from C++

可以使用 `QQmlComponent` 或 `QQuickView` 加载 QML 文档。`QQmlComponent` 将 QML 文档加载为一个 C++ 对象，然后可以从 C++ 代码进行修改。`QQuickView` 也能做到这一点，但是由于 `QQuickView` 是派生自 `QWindow` 的类，因此加载的对象也将被渲染成视觉显示；`QQuickView` 通常用于将可显示的 QML 对象集成到应用程序的的用户界面中。

例如，假设有一个名为 `MyItem.qml` 的文件，如下所示：

```javascript
import QtQuick

Item {
    width: 100; height: 100
}
```

可以使用以下 C++ 代码通过 `QQmlComponent` 或 `QQuickView` 加载此 QML 文档。使用 `QQmlComponent` 需要调用 `QQmlComponent::create()` 来创建一个组件的新实例，而 `QQuickView` 会自动创建一个组件实例，可以通过 `QQuickView::rootObject()` 访问它：

```javascript
// Using QQmlComponent
QQmlEngine engine;
QQmlComponent component(&engine,
        QUrl::fromLocalFile("MyItem.qml"));
QObject *object = component.create();
...
delete object;
```

```javascript
// Using QQuickView
QQuickView view;
view.setSource(QUrl::fromLocalFile("MyItem.qml"));
view.show();
QObject *object = view.rootObject();
```

`object` 是创建的 `MyItem.qml` 组件的实例。现在可以使用 `QObject::setProperty()` 或 `QQmlProperty::write()` 修改该项目的属性：

```javascript
object->setProperty("width", 500);
QQmlProperty(object, "width").write(500);
```

`QObject::setProperty()` 和 `QQmlProperty::write()` 的区别在于，后者除了设置属性值之外还会移除绑定。例如，如果上面的 `width` 赋值是一个绑定到 `height` 的绑定：

```javascript
width: height
```

那么在 `object->setProperty("width", 500)` 调用之后，如果 `Item` 的 `height` 发生变化，则 `width` 将再次更新，因为绑定仍然处于活动状态。但是，如果在 `QQmlProperty(object, "width").write(500)` 调用之后 `height` 发生变化，则 `width` 将不会改变，因为绑定不再存在。

或者，您可以将对象转换为其实际类型并调用具有编译时安全性的方法。在这种情况下，`MyItem.qml` 的基对象是一个 `Item`，它由 `QQuickItem` 类定义：

```javascript
QQuickItem *item = qobject_cast<QQuickItem*>(object);
item->setWidth(500);
```

您还可以使用 `QMetaObject::invokeMethod()` 和 `QObject::connect()` 连接到组件中定义的任何信号或调用方法。有关详细信息，请参见下文的调用 QML 方法和连接到 QML 信号部分。

## Accessing QML Objects via well-defined C++ Interfaces

通过在 C++ 中定义一个接口并在 QML 中访问它，是 C++ 与 QML 交互的最佳方式。使用其他方法，重构您的 QML 代码很容易导致您的 QML/C++ 交互中断。它还有助于推理 QML 和 C++ 代码的交互，因为通过 QML 驱动它可以让用户和诸如 qmllint 之类的工具更轻松地进行推理。

从 C++ 访问 QML 将导致无法理解的 QML 代码，除非手动验证没有外部 C++ 代码修改给定的 QML 组件，即使这样，访问的范围也可能会随着时间的推移而改变，使继续使用这种策略成为维护负担。

为了让 QML 驱动交互，首先需要定义一个 C++ 接口：

```c++
class CppInterface : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    // ...
};
```

使用 QML 驱动的这种方法，可以通过两种方式与该接口进行交互：

### Singletons

一种选择是通过将 `QML_SINGLETON` 宏添加到接口来注册该接口为单例，从而将其暴露给所有组件。然后，可以通过简单的 import 语句使用该接口：

```javascript
import my.company.module

Item {
    Component.onCompleted: {
        CppInterface.foo();
    }
}
```

如果您需要在比 root 组件更多的地方使用您的接口，请使用这种方法，因为简单地传递一个对象需要通过属性显式地将其传递给其他组件，或者使用不推荐的、速度较慢的未限定访问方法。

### Initial properties

另一种选择是通过 `QML_UNCREATABLE` 将接口标记为不可创建，并通过使用 `QQmlComponent::createWithInitialProperties()` 和 QML 端的必需属性将其提供给根 QML 组件。

您的 root 组件可能如下所示：

```javascript
import QtQuick

Item {
    required property CppInterface interface
    Component.onCompleted: {
        interface.foo();
    }
}
```

此处将属性标记为必需项可以防止在未设置接口属性的情况下创建组件。

然后，您可以使用与[从 C++ 加载 QML 对象](#Loading QML Objects from C++)中概述相同的方式初始化您的组件，除了使用 `createWithInitialProperties()`：

```javascript
component.createWithInitialProperties(QVariantMap{{u"interface"_s, QVariant::fromValue<CppInterface *>(new CppInterface)}});
```

如果您知道您的接口只需要 root 组件可以使用，那么这种方法更可取。它还允许在 C++ 端更轻松地连接到接口的信号和槽（slots）。

如果上述方法都不适合您的需求，您可能需要改用 C++ 模型。

## Accessing Loaded QML Objects by Object Name

QML 组件本质上是对象树，子对象具有兄弟姐妹及其自己的子对象。可以使用 `QObject::findChild()` 和 `QObject::objectName` 属性定位 QML 组件的子对象。例如，如果 `MyItem.qml` 中的 root 项具有子 `Rectangle` 项：

```javascript
import QtQuick

Item {
    width: 100; height: 100

    Rectangle {
        anchors.fill: parent
        objectName: "rect"
    }
}
```

子项可以这样定位：

```c++
QObject *rect = object->findChild<QObject*>("rect");
if (rect)
    rect->setProperty("color", "red");
```

请注意，一个对象可能有多个具有相同 `objectName` 的子对象。例如，`ListView` 会创建其委托的多个实例，因此，如果其委托声明了特定的 `objectName`，则 `ListView` 将具有多个具有相同 `objectName` 的子对象。在这种情况下，可以使用 `QObject::findChildren()` 找到所有具有匹配 `objectName` 的子对象。

> 警告：尽管可以从 C++ 访问 QML 对象并对其进行操作，但这并不是推荐的方法，除非用于测试和原型制作目的。QML 和 C++ 集成的一大优势在于能够将 UI 在 QML 中单独实现，使其与 C++ 逻辑和数据集后端分开，如果 C++ 端开始直接操纵 QML，则会破坏这一点。这种方法还会使更改 QML UI 而不影响其 C++ 对应部分变得困难。

## Accessing Members of a QML Object Type from C++

### Properties

在 QML 对象中声明的任何属性都可以从 C++ 自动访问。给定如下 QML 项：

```javascript
// MyItem.qml
import QtQuick

Item {
    property int someNumber: 100
}
```

可以使用 `QQmlProperty` 或 `QObject::setProperty()` 和 `QObject::property()` 设置和读取 `someNumber` 属性的值：

```c++
QQmlEngine engine;
QQmlComponent component(&engine, "MyItem.qml");
QObject *object = component.create();

qDebug() << "Property value:" << QQmlProperty::read(object, "someNumber").toInt();
QQmlProperty::write(object, "someNumber", 5000);

qDebug() << "Property value:" << object->property("someNumber").toInt();
object->setProperty("someNumber", 100);
```

您应该始终使用 `QObject::setProperty()`, `QQmlProperty` 或 `QMetaProperty::write()` 来更改 QML 属性值，以确保 QML 引擎意识到属性的更改。例如，假设您有一个自定义类型 `PushButton`，它具有一个 `buttonText` 属性，该属性在内部反映成员变量 `m_buttonText` 的值。像这样直接修改成员变量并不是一个好主意：

```c++
//bad code
QQmlComponent component(engine, "MyButton.qml");
PushButton *button = qobject_cast<PushButton*>(component.create());
button->m_buttonText = "Click me";
```

由于值被直接更改，这会绕过 Qt 的元对象系统，并且 QML 引擎不会意识到属性的更改。这意味着绑定到 `buttonText` 的属性不会被更新，并且任何 `onButtonTextChanged` 处理程序都不会被调用。

### Invoking QML Methods

所有 QML 方法都暴露给元对象系统，可以使用 `QMetaObject::invokeMethod()` 从 C++ 调用。您可以在冒号字符后指定参数和返回值的类型，如下面的代码片段所示。例如，当您想要将 C++ 中具有特定签名的信号连接到 QML 定义的方法时，这可能很有用。如果您省略类型，C++ 签名将使用 `QVariant`。

以下是如何使用 `QMetaObject::invokeMethod()` 调用 QML 方法的 C++ 应用：

QML：

```javascript
// MyItem.qml
import QtQuick

Item {
    function myQmlFunction(msg: string) : string {
        console.log("Got message:", msg)
        return "some return value"
    }
}
```

C++：

```c++
// main.cpp
QQmlEngine engine;
QQmlComponent component(&engine, "MyItem.qml");
QObject *object = component.create();

QString returnedValue;
QString msg = "Hello from C++";
QMetaObject::invokeMethod(object, "myQmlFunction",
        Q_RETURN_ARG(QString, returnedValue),
        Q_ARG(QString, msg));

qDebug() << "QML function returned:" << returnedValue;
delete object;
```

请注意冒号后指定的参数和返回类型。您可以使用值类型和对象类型作为类型名称。

如果类型在 QML 中被省略或指定为 `var`，那么在调用 `QMetaObject::invokeMethod` 时，您必须将 `QVariant` 作为类型传递，并使用 `Q_RETURN_ARG()` 和 `Q_ARG()`。

### Connecting to QML Signals

所有 QML 信号都自动可供 C++ 使用，并且可以使用 `QObject::connect()` 像任何普通的 Qt C++ 信号一样进行连接。作为回报，任何 C++ 信号都可以使用信号处理程序被 QML 对象接收。

这是一个具有名为 `qmlSignal` 的信号的 QML 组件，该信号会发射一个字符串类型的参数。该信号使用 `QObject::connect()` 连接到 C++ 对象的槽，因此每当发射 `qmlSignal` 时都会调用 `cppSlot()` 方法：

```javascript
// MyItem.qml
import QtQuick

Item {
    id: item
    width: 100; height: 100

    signal qmlSignal(msg: string)

    MouseArea {
        anchors.fill: parent
        onClicked: item.qmlSignal("Hello from QML")
    }
}
```

```c++
class MyClass : public QObject
{
    Q_OBJECT
public slots:
    void cppSlot(const QString &msg) {
        qDebug() << "Called the C++ slot with message:" << msg;
    }
};

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);

    QQuickView view(QUrl::fromLocalFile("MyItem.qml"));
    QObject *item = view.rootObject();

    MyClass myClass;
    QObject::connect(item, SIGNAL(qmlSignal(QString)),
                     &myClass, SLOT(cppSlot(QString)));

    view.show();
    return app.exec();
}
```

信号参数中的 QML 对象类型会转换为 C++ 中指向该类的指针：

```javascript
// MyItem.qml
import QtQuick 2.0

Item {
    id: item
    width: 100; height: 100

    signal qmlSignal(anObject: Item)

    MouseArea {
        anchors.fill: parent
        onClicked: item.qmlSignal(item)
    }
}
```

```c++
class MyClass : public QObject
{
    Q_OBJECT
public slots:
    void cppSlot(QQuickItem *item) {
       qDebug() << "Called the C++ slot with item:" << item;

       qDebug() << "Item dimensions:" << item->width()
                << item->height();
    }
};

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);

    QQuickView view(QUrl::fromLocalFile("MyItem.qml"));
    QObject *item = view.rootObject();

    MyClass myClass;
    QObject::connect(item, SIGNAL(qmlSignal(QVariant)),
                     &myClass, SLOT(cppSlot(QVariant)));

    view.show();
    return app.exec();
}
```



<!-- 完成标志, 看不到, 请忽略! -->

