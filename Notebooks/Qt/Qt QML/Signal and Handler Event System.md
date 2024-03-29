# [Signal and Handler Event System](https://doc.qt.io/qt-6/qtqml-syntax-signals.html)

应用程序和用户界面组件需要彼此通信。例如，按钮需要知道用户是否已单击它。按钮可以改变颜色以指示其状态或执行一些逻辑。同样，应用程序也需要知道用户是否正在单击按钮。应用程序可能需要将此单击事件转发到其他应用程序。

QML 具有信号和处理程序机制，其中信号 (signal) 是事件，通过信号处理程序 (signal handler) 响应信号。当发出信号时，将调用相应的信号处理程序。将诸如脚本或其他操作之类的逻辑放在处理程序中，可以使组件响应事件。

## Receiving signals with signal handlers

要接收特定对象发出特定信号时的通知，对象定义应声明一个名为 `on<Signal>` 的信号处理程序，其中 `<Signal>` 是信号的名称，首字母大写。信号处理程序应包含在调用信号处理程序时要执行的 JavaScript 代码。

例如，来自 [Qt Quick Controls](https://doc.qt.io/qt-6/qtquickcontrols-index.html) 模块的 `Button` 类型具有一个 `clicked` 信号，该信号会在每次单击按钮时发出。在这种情况下，用于接收此信号的信号处理程序应为 `onClicked`。在下例中，每当单击按钮时，都会调用 `onClicked` 处理程序，将随机颜色应用于父级 `Rectangle`：

```javascript
import QtQuick
import QtQuick.Controls

Rectangle {
    id: rect
    width: 250; height: 250

    Button {
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        text: "Change color!"
        onClicked: {
            rect.color = Qt.rgba(Math.random(), Math.random(), Math.random(), 1);
        }
    }
}
```

### Property change signal handlers

当 QML 属性的值更改时，会自动发出信号。此类信号是属性更改信号，这些信号的信号处理程序采用 `on<Property>Changed` 形式编写，其中 `<Property>` 是属性的名称，首字母大写。

例如，`MouseArea` 类型具有一个 `pressed` 属性。要接收此属性每次更改时的通知，请编写名为 `onPressedChanged` 的信号处理程序：

```javascript
import QtQuick

Rectangle {
    id: rect
    width: 100; height: 100

    TapHandler {
        onPressedChanged: console.log("taphandler pressed?", pressed)
    }
}
```

即使 `TapHandler` 文档没有记录名为 `onPressedChanged` 的信号处理程序，但由于存在 `pressed` 属性，该信号也会隐式提供。

### Signal parameters

信号可能具有参数。要访问它们，您应该将一个函数分配给处理程序。箭头函数和匿名函数都适用。

以下面的示例为例，考虑一个带有 `errorOccurred` 信号的 `Status` 组件（有关如何将信号添加到 QML 组件的更多信息，请参阅[向自定义 QML 类型添加信号](#Adding signals to custom QML types)）。

```javascript
// Status.qml
import QtQuick

Item {
    id: myitem
    signal errorOccurred(message: string, line: int, column: int)
}
```

> 注意：函数中的形式参数名称不必与信号中的名称匹配。

如果您不需要处理所有参数，则可以省略尾部的参数：

```javascript
Status {
    onErrorOccurred: function (message) { console.log(message) }
}
```

您不能省略您感兴趣的前导参数，但是您可以使用一些占位符名称来向读者表明它们并不重要：

```javascript
Status {
    onErrorOccurred: (_, _, col) => console.log(`Error happened at column ${col}`)
}
```

> 注意：可以使用纯代码块代替函数，但这并不鼓励。在这种情况下，所有信号参数都会被注入到块的范围内。但是，这会使代码难以阅读，因为不清楚参数来自何处，并且会导致 QML 引擎中的查找速度变慢。以这种方式注入参数已被弃用，如果实际使用了参数，将导致运行时警告。

### Using the Connections type

在某些情况下，可能需要在发出信号的对象之外访问信号。为此，`QtQuick` 模块提供了 [`Connections`](https://doc.qt.io/qt-6/qml-qtqml-connections.html) 类型，用于连接到任意对象的信号。`Connections` 对象可以从其指定的 `target` 接收任何信号。

例如，先前示例中的 `onClicked` 处理程序也可以由 `Rectangle` 接收，方法是将 `onClicked` 处理程序放置在一个 `Connections` 对象中，该对象的 `target` 设置为按钮：

```javascript
import QtQuick
import QtQuick.Controls

Rectangle {
    id: rect
    width: 250; height: 250

    Button {
        id: button
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        text: "Change color!"
    }

    Connections {
        target: button
        function onClicked() {
            rect.color = Qt.rgba(Math.random(), Math.random(), Math.random(), 1);
        }
    }
}
```

### Attached signal handlers

[附加信号处理程序](https://doc.qt.io/qt-6/qtqml-syntax-objectattributes.html#attached-properties-and-attached-signal-handlers)接收来自附加类型的信号，而不是声明处理程序的对象内的信号。

例如，[`Component.onCompleted`](https://doc.qt.io/qt-6/qml-qtqml-component.html#completed-signal) 是一个附加信号处理程序。它通常用于在其创建过程完成后执行一些 JavaScript 代码。这里是一个例子：

```javascript
import QtQuick

Rectangle {
    width: 200; height: 200
    color: Qt.rgba(Qt.random(), Qt.random(), Qt.random(), 1)

    Component.onCompleted: {
        console.log("The rectangle's color is", color)
    }
}
```

`onCompleted` 处理器并不会响应来自 `Rectangle` 的 `completed` 信号。相反，QML 引擎会自动将一个带有 `completed` 信号的 `Component.attaching` 类型对象附加到 `Rectangle` 对象上。引擎会在创建 `Rectangle` 对象时发出此信号，从而触发 `Component.onCompleted` 信号处理程序。

附加信号处理程序允许对象被通知与每个单独对象相关的特定信号。例如，如果没有附加的 `Component.onCompleted` 信号处理程序，对象就无法接收此通知，除非它从某个特殊对象注册某些特殊信号。附加信号处理程序机制使对象无需额外代码即可接收特定信号。

有关附加信号处理程序的更多信息，请参见[附加属性和附加信号处理程序](https://doc.qt.io/qt-6/qtqml-syntax-objectattributes.html#attached-properties-and-attached-signal-handlers)。

## Adding signals to custom QML types

可以通过 `signal` 关键字将信号添加到自定义 QML 类型。

定义新信号的语法为：

```javascript
signal <name>[([<type> <parameter name>[, ...]])]
```

通过将信号作为方法调用来发出信号。

例如，以下代码定义在名为 `SquareButton.qml` 的文件中。root `Rectangle` 对象具有一个 `activated` 信号，每当子 `TapHandler` 被点击时就会发出该信号。在这个特定示例中，`activated` 信号会随着鼠标点击的 x 和 y 坐标一起发出：

```javascript
// SquareButton.qml
import QtQuick

Rectangle {
    id: root

    signal activated(real xPosition, real yPosition)
    property point mouseXY
    property int side: 100
    width: side; height: side

    TapHandler {
        id: handler
        onTapped: root.activated(root.mouseXY.x, root.mouseXY.y)
        onPressedChanged: root.mouseXY = handler.point.position
    }
}
```

现在任何 `SquareButton` 对象都可以使用 `onActivated` 信号处理程序连接到 `activated` 信号：

```javascript
// myapplication.qml
SquareButton {
    onActivated: (xPosition, yPosition)=> console.log("Activated at " + xPosition + "," + yPosition)
}
```

有关为自定义 QML 类型编写信号的更多详细信息，请参见[信号属性](https://doc.qt.io/qt-6/qtqml-syntax-objectattributes.html#signal-attributes)。

## Connecting signals to methods and signals

信号对象有一个 `connect()` 方法，可以将信号连接到方法或另一个信号。当信号连接到方法时，每当信号发出时，该方法都会自动调用。这种机制使信号可以通过方法而不是信号处理程序来接收。

下面，使用 `connect()` 方法将 `messageReceived` 信号连接到三个方法：

```javascript
import QtQuick

Rectangle {
    id: relay

    signal messageReceived(string person, string notice)

    Component.onCompleted: {
        relay.messageReceived.connect(sendToPost)
        relay.messageReceived.connect(sendToTelegraph)
        relay.messageReceived.connect(sendToEmail)
        relay.messageReceived("Tom", "Happy Birthday")
    }

    function sendToPost(person, notice) {
        console.log("Sending to post: " + person + ", " + notice)
    }
    function sendToTelegraph(person, notice) {
        console.log("Sending to telegraph: " + person + ", " + notice)
    }
    function sendToEmail(person, notice) {
        console.log("Sending to email: " + person + ", " + notice)
    }
}
```

在许多情况下，通过信号处理程序接收信号而不是使用 `connect()` 函数就足够了。但是，使用 `connect` 方法允许一个信号被多个方法接收，如前所述，这使用信号处理程序 (`on<Signal>` ) 是不可能的，因为它们必须具有唯一名称。此外，`connect` 方法在将信号连接到动态创建的对象时也很有用。

还有一个对应的 `disconnect()` 方法用于移除已连接的信号。

### Signal to signal connect

通过将信号连接到其他信号，`connect()` 方法可以形成不同的信号链。例如：

```javascript
import QtQuick

Rectangle {
    id: forwarder
    width: 100; height: 100

    signal send()
    onSend: console.log("Send clicked")

    TapHandler {
        id: mousearea
        anchors.fill: parent
        onTapped: console.log("Mouse clicked")
    }

    Component.onCompleted: {
        mousearea.tapped.connect(send)
    }
}
```

每当发出 `TapHandler` 的 `tapped` 信号时，`send` 信号也将自动发出。

```bash
output:
    MouseArea clicked
    Send clicked
```



<!-- 完成标志, 看不到, 请忽略! -->
